"""Honest evaluation: time-ordered (walk-forward) splits and proper scoring rules.

Usage:
    python -m footy.evaluate                 # score all models on the validation seasons
    python -m footy.evaluate --split test    # final scores on the test seasons
"""

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from footy import config
from footy.baselines import all_baselines
from footy.data import OUTCOMES, load_matches

EPS = 1e-15  # floor for probabilities, so log(0) can't produce infinity
REPORTS_DIR = config.ROOT / "reports"


# ---------- Metrics ----------
# All take `probs` as an (n_matches, 3) array in H, D, A order, and `results` as 'H'/'D'/'A' labels.


def one_hot(results) -> np.ndarray:
    """'H' -> [1, 0, 0], 'D' -> [0, 1, 0], 'A' -> [0, 0, 1]."""
    results = np.asarray(results)
    return (results[:, None] == np.array(OUTCOMES)[None, :]).astype(float)


def check_probs(probs, n: int) -> np.ndarray:
    probs = np.asarray(probs, dtype=float)
    if probs.shape != (n, 3):
        raise ValueError(f"expected shape ({n}, 3), got {probs.shape}")
    if np.isnan(probs).any() or (probs < 0).any():
        raise ValueError("probabilities must be non-negative numbers")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("each row of probabilities must sum to 1")
    return probs


def log_losses(probs, results) -> np.ndarray:
    """Per-match log loss: -log(probability given to what actually happened).

    0 = gave the actual result 100%. Being confidently wrong is punished hard:
    giving the actual result 1% costs 4.6, giving it 50% costs 0.69.
    """
    probs = check_probs(probs, len(results))
    p_actual = (probs * one_hot(results)).sum(axis=1)
    return -np.log(np.clip(p_actual, EPS, 1.0))


def log_loss(probs, results) -> float:
    return float(log_losses(probs, results).mean())


def brier_score(probs, results) -> float:
    """Mean squared error between probabilities and the one-hot result, summed over H/D/A.

    0 = perfect, 2 = certain and wrong. Uniform (1/3 each) always scores 2/3.
    """
    probs = check_probs(probs, len(results))
    return float(((probs - one_hot(results)) ** 2).sum(axis=1).mean())


def accuracy(probs, results) -> float:
    """Share of matches where the most likely outcome happened. Ignores confidence: secondary only.

    Ties go to the first outcome (home), so uniform scores the home-win rate.
    """
    probs = check_probs(probs, len(results))
    predicted = np.array(OUTCOMES)[probs.argmax(axis=1)]
    return float((predicted == np.asarray(results)).mean())


# ---------- Time-ordered splits ----------


def walk_forward_splits(
    df: pd.DataFrame, seasons: list[str]
) -> Iterator[tuple[str, pd.DataFrame, pd.DataFrame]]:
    """For each season: train = every match played before it, test = that season.

    This copies real life: when a season starts, you only know what happened before it.
    A random train/test split would let the model learn from future matches (data leakage).
    """
    for season in seasons:
        test = df[df["season"] == season]
        if test.empty:
            raise ValueError(f"no matches for season {season}")
        train = df[df["date"] < test["date"].min()]
        assert train["date"].max() < test["date"].min(), "leakage: training data overlaps test season"
        yield season, train, test


# ---------- Running models ----------


class Model(Protocol):
    """Anything with a name and a predict() that sees only the past is a model."""

    name: str

    def predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        """Return (len(test), 3) probabilities in H, D, A order.

        `test` includes results so that models like Elo can update as the season goes on,
        but a prediction for a match may only use results from EARLIER dates.
        """
        ...


def _scores(probs: np.ndarray, results: np.ndarray) -> dict:
    losses = log_losses(probs, results)
    return {
        "matches": len(results),
        "log_loss": losses.mean(),
        "log_loss_se": losses.std(ddof=1) / np.sqrt(len(losses)),  # standard error: how noisy the estimate is
        "brier": brier_score(probs, results),
        "accuracy": accuracy(probs, results),
    }


def evaluate(models: list[Model], df: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Score every model on every season, plus 'all' = every test match pooled together."""
    rows = []
    for model in models:
        all_probs, all_results = [], []
        for season, train, test in walk_forward_splits(df, seasons):
            probs = check_probs(model.predict(train, test), len(test))
            results = test["result"].to_numpy()
            rows.append({"model": model.name, "season": season, **_scores(probs, results)})
            all_probs.append(probs)
            all_results.append(results)
        pooled = _scores(np.vstack(all_probs), np.concatenate(all_results))
        rows.append({"model": model.name, "season": "all", **pooled})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", choices=["validation", "test"], default="validation")
    args = p.parse_args(argv)
    seasons = config.VALIDATION_SEASONS if args.split == "validation" else config.TEST_SEASONS

    results = evaluate(all_baselines(), load_matches(), seasons)

    overall = results[results["season"] == "all"].drop(columns="season").sort_values("log_loss")
    print(f"\n{args.split.upper()} seasons {seasons[0]} to {seasons[-1]} (lower log loss / Brier = better)\n")
    print(overall.round(4).to_string(index=False))

    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"{args.split}_results.csv"
    results.round(5).to_csv(out, index=False)
    print(f"\nPer-season results saved to {Path(out).relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
