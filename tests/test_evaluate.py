import math

import numpy as np
import pandas as pd
import pytest

from footy.baselines import BaseRates, Bookmaker, Uniform
from footy.evaluate import accuracy, brier_score, evaluate, log_loss, walk_forward_splits

RESULTS = np.array(["H", "D", "A", "H"])
PERFECT = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=float)
UNIFORM = np.full((4, 3), 1 / 3)


def make_matches(seasons: dict[str, str]) -> pd.DataFrame:
    """Tiny fake dataset: {'2019-20': 'HHDA', ...} -> one match per letter, dated in order."""
    rows = []
    for season, results in seasons.items():
        start = pd.Timestamp(f"{season[:4]}-08-10")
        for i, r in enumerate(results):
            rows.append(
                {
                    "season": season,
                    "date": start + pd.Timedelta(days=7 * i),
                    "result": r,
                    "avg_home": 2.0,
                    "avg_draw": 3.5,
                    "avg_away": 4.0,
                }
            )
    return pd.DataFrame(rows)


# ---------- metrics ----------


def test_perfect_predictions_score_zero():
    assert log_loss(PERFECT, RESULTS) == pytest.approx(0.0, abs=1e-9)
    assert brier_score(PERFECT, RESULTS) == 0.0
    assert accuracy(PERFECT, RESULTS) == 1.0


def test_uniform_scores_known_values():
    assert log_loss(UNIFORM, RESULTS) == pytest.approx(math.log(3))  # 1.0986
    assert brier_score(UNIFORM, RESULTS) == pytest.approx(2 / 3)


def test_confident_and_wrong_is_punished_but_finite():
    wrong = np.array([[0, 0, 1]] * 4, dtype=float)  # 100% away win every time
    assert brier_score(wrong, ["H"] * 4) == 2.0
    loss = log_loss(wrong, ["H"] * 4)
    assert 30 < loss < 40 and math.isfinite(loss)  # capped by the 1e-15 floor


def test_bad_probabilities_rejected():
    with pytest.raises(ValueError, match="sum to 1"):
        log_loss(np.full((4, 3), 0.5), RESULTS)
    with pytest.raises(ValueError, match="shape"):
        log_loss(np.full((3, 3), 1 / 3), RESULTS)


# ---------- splits ----------


def test_walk_forward_never_trains_on_the_future():
    df = make_matches({"2018-19": "HHDA", "2019-20": "HDAA", "2020-21": "AAHD"})
    splits = list(walk_forward_splits(df, ["2019-20", "2020-21"]))
    assert [s for s, _, _ in splits] == ["2019-20", "2020-21"]
    for _, train, test in splits:
        assert train["date"].max() < test["date"].min()
    _, train_2021, _ = splits[1]
    assert set(train_2021["season"]) == {"2018-19", "2019-20"}  # grows each season


def test_unknown_season_raises():
    with pytest.raises(ValueError, match="no matches"):
        list(walk_forward_splits(make_matches({"2018-19": "HD"}), ["1999-00"]))


# ---------- baselines + evaluate ----------


def test_base_rates_use_training_data_only():
    df = make_matches({"2018-19": "HHHD", "2019-20": "AAAA"})
    _, train, test = next(walk_forward_splits(df, ["2019-20"]))
    probs = BaseRates().predict(train, test)
    # learned 3/4 home, 1/4 draw from 2018-19, and is NOT influenced by the all-away test season
    assert probs[0] == pytest.approx([0.75, 0.25, 0.0])


def test_base_rates_recent_window():
    df = make_matches({"2017-18": "AAAA", "2018-19": "HHHH", "2019-20": "DDDD"})
    _, train, test = next(walk_forward_splits(df, ["2019-20"]))
    assert BaseRates(recent_seasons=1).predict(train, test)[0] == pytest.approx([1, 0, 0])


def test_bookmaker_probabilities_sum_to_one():
    df = make_matches({"2019-20": "HDA"})
    probs = Bookmaker().predict(df.iloc[:0], df)
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])


def test_evaluate_reports_each_season_and_pooled():
    df = make_matches({"2018-19": "HHDA", "2019-20": "HDAH", "2020-21": "AHHD"})
    out = evaluate([Uniform(), BaseRates()], df, ["2019-20", "2020-21"])
    assert set(out["season"]) == {"2019-20", "2020-21", "all"}
    uniform_all = out[(out["model"] == "uniform") & (out["season"] == "all")].iloc[0]
    assert uniform_all["matches"] == 8
    assert uniform_all["log_loss"] == pytest.approx(math.log(3))
