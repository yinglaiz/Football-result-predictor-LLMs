from pathlib import Path

import pandas as pd
import pytest

from footy.data import implied_probabilities, load_matches
from footy.db import connect
from footy.ingest import ingest_file

DATA = Path(__file__).parent / "data"


def test_implied_probabilities_remove_margin():
    odds = pd.DataFrame({"avg_home": [2.0, 1.30], "avg_draw": [3.5, 6.0], "avg_away": [4.0, 9.5]})
    p = implied_probabilities(odds)
    # each row sums to exactly 1 once the margin is removed
    assert (p[["p_home", "p_draw", "p_away"]].sum(axis=1)).tolist() == pytest.approx([1.0, 1.0])
    # 1/2 + 1/3.5 + 1/4 = 1.0357 -> margin 3.57%
    assert p.loc[0, "margin"] == pytest.approx(0.0357, abs=1e-4)
    # favourite stays the favourite
    assert p.loc[1, "p_home"] > p.loc[1, "p_draw"] > p.loc[1, "p_away"]


def test_load_matches_sorted_with_datetimes(tmp_path):
    db = tmp_path / "t.db"
    conn = connect(db)
    ingest_file(conn, DATA / "E0_2526_sample.csv", "E0", "2025-26")  # loaded first on purpose
    ingest_file(conn, DATA / "E0_0506_sample.csv", "E0", "2005-06")
    conn.close()

    df = load_matches(db)
    assert len(df) == 6
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["date"].is_monotonic_increasing  # oldest first, whatever the load order


def test_load_matches_missing_db_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="footy.ingest"):
        load_matches(tmp_path / "nope.db")
