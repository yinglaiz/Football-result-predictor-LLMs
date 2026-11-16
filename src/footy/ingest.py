"""Download historical match results from football-data.co.uk and load them into SQLite.

Usage:
    python -m footy.ingest                 # all configured seasons (cached files reused)
    python -m footy.ingest --start 2020    # only 2020/21 onwards
    python -m footy.ingest --refresh       # force re-download of every season
"""

import argparse
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import requests

from footy import config
from footy.db import connect

log = logging.getLogger(__name__)

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Source column -> database column. Odds columns changed name over the years,
# so each target lists candidates in order of preference.
STATS = {"HS": "home_shots", "AS": "away_shots", "HST": "home_shots_target", "AST": "away_shots_target"}
ODDS = {
    "b365_home": ["B365H"],
    "b365_draw": ["B365D"],
    "b365_away": ["B365A"],
    "avg_home": ["AvgH", "BbAvH"],
    "avg_draw": ["AvgD", "BbAvD"],
    "avg_away": ["AvgA", "BbAvA"],
}
DB_COLUMNS = [
    "match_id",
    "league",
    "season",
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    *STATS.values(),
    *ODDS.keys(),
]


# ---------- 1. Download (with a local cache) ----------


def download_season(league: str, start_year: int, refresh: bool = False) -> Path:
    """Return the path of the cached CSV, downloading it if needed.

    Finished seasons never change, so they are downloaded once. The current season is
    always re-downloaded because new results are added every week.
    """
    path = config.RAW_DIR / f"{league}_{config.season_code(start_year)}.csv"
    is_current = start_year >= config.CURRENT_SEASON_START
    if path.exists() and not refresh and not is_current:
        return path

    url = config.BASE_URL.format(season=config.season_code(start_year), league=league)
    log.info("Downloading %s", url)
    resp = requests.get(url, headers={"User-Agent": config.USER_AGENT}, timeout=30)
    resp.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(resp.content)
    time.sleep(config.REQUEST_DELAY_S)
    return path


# ---------- 2. Parse + clean ----------


def _read_csv(path: Path) -> pd.DataFrame:
    # Older files are not UTF-8; trailing commas create empty "Unnamed" columns.
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", on_bad_lines="warn")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1", on_bad_lines="warn")
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _parse_dates(s: pd.Series) -> pd.Series:
    """Dates are dd/mm/yyyy in recent files and dd/mm/yy in older ones."""
    s = s.astype(str).str.strip()
    long = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
    short = pd.to_datetime(s, format="%d/%m/%y", errors="coerce")
    dates = long.fillna(short)
    if dates.isna().any():
        bad = s[dates.isna()].unique()[:5]
        raise ValueError(f"Unparseable dates: {list(bad)}")
    return dates.dt.strftime("%Y-%m-%d")


def parse_season_csv(path: Path, league: str, season: str) -> pd.DataFrame:
    raw = _read_csv(path)
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}")

    # Drop blank trailing rows and fixtures without a full-time score yet
    raw = raw.dropna(subset=["HomeTeam", "AwayTeam"])
    unplayed = raw[["FTHG", "FTAG", "FTR"]].isna().any(axis=1)
    if unplayed.any():
        log.warning("%s: skipping %d rows with no result", path.name, int(unplayed.sum()))
    raw = raw[~unplayed]

    df = pd.DataFrame(
        {
            "league": league,
            "season": season,
            "date": _parse_dates(raw["Date"]),
            "home_team": raw["HomeTeam"].str.strip(),
            "away_team": raw["AwayTeam"].str.strip(),
            "home_goals": raw["FTHG"].astype(int),
            "away_goals": raw["FTAG"].astype(int),
            "result": raw["FTR"].str.strip(),
        }
    )
    for src, dst in STATS.items():
        df[dst] = pd.to_numeric(raw[src], errors="coerce").astype("Int64") if src in raw else pd.NA
    for dst, candidates in ODDS.items():
        src = next((c for c in candidates if c in raw.columns), None)
        df[dst] = pd.to_numeric(raw[src], errors="coerce") if src else float("nan")

    df["match_id"] = (
        df["league"]
        + "_"
        + df["date"]
        + "_"
        + df["home_team"].str.replace(" ", "-")
        + "_"
        + df["away_team"].str.replace(" ", "-")
    )
    return df[DB_COLUMNS].reset_index(drop=True)


# ---------- 3. Validate ----------


def validate(df: pd.DataFrame) -> None:
    """Fail loudly on bad data rather than silently training on it."""
    problems = []
    if df["match_id"].duplicated().any():
        problems.append(f"duplicate matches: {df.loc[df['match_id'].duplicated(), 'match_id'].tolist()[:3]}")
    if (df[["home_goals", "away_goals"]] < 0).any().any():
        problems.append("negative goals")
    expected = pd.Series("D", index=df.index)
    expected[df["home_goals"] > df["away_goals"]] = "H"
    expected[df["home_goals"] < df["away_goals"]] = "A"
    if (expected != df["result"]).any():
        problems.append(f"{int((expected != df['result']).sum())} results inconsistent with score")
    odds = df[list(ODDS)]
    if (odds.notna() & (odds <= 1.0)).any().any():
        problems.append("decimal odds <= 1.0")
    if (df["home_team"] == df["away_team"]).any():
        problems.append("team playing itself")
    if problems:
        raise ValueError("Validation failed: " + "; ".join(problems))


# ---------- 4. Load (upsert) ----------


def upsert(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """Insert new matches, update existing ones. Re-running never creates duplicates."""
    cols = DB_COLUMNS
    updates = ", ".join(f"{c} = excluded.{c}" for c in cols if c != "match_id")
    sql = (
        f"INSERT INTO matches ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
        f"ON CONFLICT(match_id) DO UPDATE SET {updates}, loaded_at = datetime('now')"
    )
    # Convert pandas missing values (NaN / <NA>) to None, which SQLite stores as NULL
    rows = [
        tuple(None if pd.isna(v) else (v.item() if hasattr(v, "item") else v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]
    with conn:  # one transaction: all rows or none
        conn.executemany(sql, rows)
    return len(rows)


def ingest_file(conn: sqlite3.Connection, path: Path, league: str, season: str) -> int:
    df = parse_season_csv(path, league, season)
    validate(df)
    return upsert(conn, df)


# ---------- CLI ----------


def summary(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(
        "SELECT league, season, COUNT(*) AS matches, MIN(date) AS first, MAX(date) AS last, "
        "ROUND(AVG(result = 'H'), 3) AS home_win_rate, "
        "ROUND(AVG(avg_home IS NOT NULL), 2) AS odds_coverage "
        "FROM matches GROUP BY league, season ORDER BY league, season",
        conn,
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--leagues", nargs="+", default=list(config.LEAGUES))
    p.add_argument("--start", type=int, default=config.FIRST_SEASON_START, help="first season start year")
    p.add_argument("--end", type=int, default=config.CURRENT_SEASON_START, help="last season start year")
    p.add_argument("--refresh", action="store_true", help="re-download cached seasons")
    p.add_argument("--db", default=str(config.DB_PATH))
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    conn = connect(args.db)
    total = 0
    for league in args.leagues:
        for year in range(args.start, args.end + 1):
            path = download_season(league, year, refresh=args.refresh)
            n = ingest_file(conn, path, league, config.season_label(year))
            log.info("%s %s: %d matches", league, config.season_label(year), n)
            total += n
    log.info("Loaded %d matches into %s", total, args.db)
    with pd.option_context("display.width", 120):
        print(summary(conn).to_string(index=False))
    conn.close()


if __name__ == "__main__":
    main()
