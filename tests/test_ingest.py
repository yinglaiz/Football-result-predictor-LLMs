"""Tests for the ingestion pipeline. They use small sample files, so they run offline in < 1 s."""

from pathlib import Path

import pandas as pd
import pytest

from footy.config import season_code, season_label
from footy.db import connect
from footy.ingest import ingest_file, parse_season_csv, summary, validate

DATA = Path(__file__).parent / "data"
OLD = DATA / "E0_0506_sample.csv"  # dd/mm/yy dates, 'BbAv' odds columns, trailing blank row
NEW = DATA / "E0_2526_sample.csv"  # dd/mm/yyyy dates, 'Avg' odds columns, one unplayed fixture


def test_season_helpers():
    assert season_code(2005) == "0506"
    assert season_code(1999) == "9900"
    assert season_label(2025) == "2025-26"


def test_parse_old_format():
    df = parse_season_csv(OLD, "E0", "2005-06")
    assert len(df) == 3  # blank row dropped
    assert df.loc[0, "date"] == "2005-08-13"
    assert df.loc[1, "avg_away"] == pytest.approx(1.73)  # read from BbAvA
    assert df.loc[2, "result"] == "H"


def test_parse_new_format_skips_unplayed():
    df = parse_season_csv(NEW, "E0", "2025-26")
    assert len(df) == 3  # Brentford v Sunderland has no result yet
    assert df.loc[0, "match_id"] == "E0_2025-08-15_Liverpool_Bournemouth"
    assert df.loc[2, "avg_home"] == pytest.approx(3.45)  # read from AvgH (not closing AvgCH)
    assert df["home_shots_target"].tolist() == [10, 1, 6]


def test_validate_catches_inconsistent_result():
    df = parse_season_csv(NEW, "E0", "2025-26")
    df.loc[0, "result"] = "A"  # Liverpool won 4-2, so 'A' is wrong
    with pytest.raises(ValueError, match="inconsistent"):
        validate(df)


def test_validate_catches_bad_odds():
    df = parse_season_csv(NEW, "E0", "2025-26")
    df.loc[0, "b365_home"] = 0.9
    with pytest.raises(ValueError, match="odds"):
        validate(df)


def test_reloading_is_idempotent(tmp_path):
    """Running the pipeline twice must not duplicate rows."""
    conn = connect(tmp_path / "test.db")
    for _ in range(2):
        ingest_file(conn, OLD, "E0", "2005-06")
        ingest_file(conn, NEW, "E0", "2025-26")
    n = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    assert n == 6

    s = summary(conn)
    assert s["matches"].tolist() == [3, 3]
    # missing stats stay NULL rather than becoming 0
    nulls = pd.read_sql("SELECT COUNT(*) AS n FROM matches WHERE home_shots IS NULL", conn)
    assert nulls.loc[0, "n"] == 0
    conn.close()
