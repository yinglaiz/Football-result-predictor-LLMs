"""SQLite database: schema and helpers.

SQLite stores the whole database in one file (data/football.db). No server to install,
and Python ships with the `sqlite3` module.
"""

import sqlite3
from pathlib import Path

from footy.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS matches (
    match_id      TEXT PRIMARY KEY,   -- league_date_home_away: stable, so reloads never duplicate rows
    league        TEXT NOT NULL,
    season        TEXT NOT NULL,      -- e.g. '2025-26'
    date          TEXT NOT NULL,      -- ISO format YYYY-MM-DD (sorts correctly as text)
    home_team     TEXT NOT NULL,
    away_team     TEXT NOT NULL,
    home_goals    INTEGER NOT NULL,
    away_goals    INTEGER NOT NULL,
    result        TEXT NOT NULL CHECK (result IN ('H', 'D', 'A')),
    -- match stats: only known AFTER kick-off. Use them for a team's PAST matches only (leakage!)
    home_shots          INTEGER,
    away_shots          INTEGER,
    home_shots_target   INTEGER,
    away_shots_target   INTEGER,
    -- pre-match decimal odds (for the bookmaker benchmark in Stage 3)
    b365_home REAL, b365_draw REAL, b365_away REAL,   -- Bet365
    avg_home  REAL, avg_draw  REAL, avg_away  REAL,   -- market average
    loaded_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_matches_date ON matches (date);
CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches (home_team, away_team);
"""


def connect(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn
