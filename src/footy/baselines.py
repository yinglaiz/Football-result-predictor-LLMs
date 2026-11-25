"""Simple baseline models. Anything smarter has to beat these to be worth having."""

import numpy as np
import pandas as pd

from footy.data import OUTCOMES, implied_probabilities


class Uniform:
    """1/3 each outcome: knows nothing. Anything scoring worse is broken."""

    name = "uniform"

    def predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        return np.full((len(test), 3), 1 / 3)


class BaseRates:
    """The historical home/draw/away split, from the training data only.

    Same prediction for every match: ignores who is playing.
    """

    name = "base_rates"

    def __init__(self, recent_seasons: int | None = None):
        self.recent_seasons = recent_seasons  # None = use all history
        if recent_seasons:
            self.name = f"base_rates_last{recent_seasons}"

    def predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        if self.recent_seasons:
            keep = train["season"].unique()[-self.recent_seasons :]
            train = train[train["season"].isin(keep)]
        rates = train["result"].value_counts(normalize=True).reindex(OUTCOMES, fill_value=0.0)
        return np.tile(rates.to_numpy(), (len(test), 1))


class Bookmaker:
    """Market-average odds with the margin removed. Not a baseline but the benchmark to aim for."""

    name = "bookmaker"

    def predict(self, train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
        probs = implied_probabilities(test)[["p_home", "p_draw", "p_away"]].to_numpy()
        if np.isnan(probs).any():
            raise ValueError("some test matches have no odds")
        return probs


def all_baselines() -> list:
    return [Uniform(), BaseRates(), BaseRates(recent_seasons=3), Bookmaker()]
