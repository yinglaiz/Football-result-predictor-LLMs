"""Load match data from the database for analysis and modelling."""

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

from footy.config import DB_PATH

# Every probability array in this project uses this column order: home win, draw, away win.
OUTCOMES = ["H", "D", "A"]


def load_matches(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """All matches, oldest first, with `date` as a real datetime."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"{db_path} not found - run `python -m footy.ingest` first")
    with closing(sqlite3.connect(db_path)) as conn:
        df = pd.read_sql("SELECT * FROM matches ORDER BY date, match_id", conn, parse_dates=["date"])
    return df.drop(columns="loaded_at")


def implied_probabilities(df: pd.DataFrame, prefix: str = "avg") -> pd.DataFrame:
    """Convert bookmaker decimal odds into probabilities.

    1/odds gives the bookmaker's probability for each outcome, but these add up to
    slightly more than 1: the extra (the "margin" or "overround") is the bookmaker's profit.
    Dividing by the total removes it so the three probabilities sum to 1.
    (This "proportional" method is the simplest; Shin's method is a more refined alternative.)
    """
    odds = df[[f"{prefix}_home", f"{prefix}_draw", f"{prefix}_away"]].to_numpy(dtype=float)
    inverse = 1.0 / odds
    total = inverse.sum(axis=1, keepdims=True)
    probs = inverse / total
    return pd.DataFrame(
        {
            "p_home": probs[:, 0],
            "p_draw": probs[:, 1],
            "p_away": probs[:, 2],
            "margin": total[:, 0] - 1.0,
        },
        index=df.index,
    )
