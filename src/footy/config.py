"""Central settings: paths, leagues and seasons to load."""

from pathlib import Path

# Project root = two levels up from this file (src/footy/config.py -> repo root)
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"  # cached CSV downloads (not committed)
DB_PATH = DATA_DIR / "football.db"  # SQLite database file (not committed)

# Source: https://www.football-data.co.uk (free CSVs; credit the source, don't redistribute raw files)
BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
USER_AGENT = "football-predictor portfolio project (personal, non-commercial)"
REQUEST_DELAY_S = 1.5  # pause between downloads to be polite to a free, one-person site

LEAGUES = {"E0": "Premier League"}  # add "E1": "Championship" etc. later for more data
FIRST_SEASON_START = 2005  # 2005/06: odds and shot stats are consistently available from here
CURRENT_SEASON_START = 2026  # 2026/27 (in progress) - bump each August


def season_code(start_year: int) -> str:
    """2025 -> '2526' (the code used in football-data.co.uk URLs)."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_label(start_year: int) -> str:
    """2025 -> '2025-26' (human-readable label stored in the database)."""
    return f"{start_year}-{(start_year + 1) % 100:02d}"
