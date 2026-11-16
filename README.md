# Football Predictor

Predicts Premier League match results (home win / draw / away win) as probabilities, tracks how accurate those predictions are over time, and publishes everything on a live dashboard.

The emphasis is on **honest evaluation**: time-ordered splits, no data leakage, proper scoring rules (log loss, Brier score), and comparison against simple baselines and bookmaker odds, reported even when the model loses.

## Status

| Stage | What | Status |
|---|---|---|
| 1 | Data ingestion: historical results → SQLite, with validation and tests | ✅ |
| 2 | Baselines (home win, base rates, Elo) with time-based evaluation | ⏳ |
| 3 | Feature engineering + gradient boosting vs. baselines vs. bookmakers | |
| 4 | FastAPI backend + Streamlit dashboard | |
| 5 | Scheduled updates, weekly retraining, prediction log | |
| 6 | Docker, CI (GitHub Actions), cloud deployment | |
| 7 | Plain-English questions via LLM text-to-SQL | |

## Quickstart

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -e ".[dev]"
python -m footy.ingest            # downloads ~21 seasons, builds data/football.db (~1 min)
python -m pytest                  # runs offline tests
```

Options: `python -m footy.ingest --start 2020` (fewer seasons), `--refresh` (re-download everything), `--leagues E0 E1` (add the Championship).

## Data

Match results, shot statistics and pre-match odds come from **[football-data.co.uk](https://www.football-data.co.uk)**, which provides them free for analysis. The site publishes no explicit redistribution licence, so this repo:

- downloads data at runtime rather than committing it (`data/` is git-ignored),
- caches finished seasons locally and waits between requests,
- credits the source here and on the dashboard.

Premier League data is loaded from **2005/06** onwards, the point from which odds and shot stats are consistently available.

### `matches` table

| Column | Notes |
|---|---|
| `match_id` | `league_date_home_away`, the primary key, so reloading never creates duplicates |
| `league`, `season`, `date` | date stored as ISO `YYYY-MM-DD` |
| `home_team`, `away_team`, `home_goals`, `away_goals`, `result` | `result` ∈ {H, D, A} |
| `home_shots`, `away_shots`, `home_shots_target`, `away_shots_target` | **post-match** stats: only ever used as features for *later* matches |
| `b365_*`, `avg_*` | pre-match decimal odds (Bet365, market average), used as a benchmark |

## Project structure

```
src/footy/
  config.py    paths, leagues, seasons
  db.py        SQLite schema + connection
  ingest.py    download → parse → validate → upsert
tests/
  data/        small sample CSVs covering old and new file formats
  test_ingest.py
```

## Design decisions

- **SQLite** for now: a single-file database with no server needed. Moving to Postgres for deployment (Stage 6) only changes the connection.
- **Validation before loading**: score/result consistency, valid odds and duplicate checks fail loudly instead of letting bad rows reach the model.
- **Idempotent upserts**: the pipeline can be re-run safely (e.g. on a weekly schedule) and only updates what changed.
- **Format drift handled explicitly**: two date formats and renamed odds columns across 20 seasons, all covered by tests.

---
*Data: football-data.co.uk. This project is for analysis and learning, not betting advice.*
