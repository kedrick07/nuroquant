# NuroQuant

Last updated: 2026-10-19

NuroQuant is a final year project (UPM, FSKTM, Dept. of Multimedia) by Kedrick Selvanesan. Supervisor: Assoc. Prof. Dr. Azreen bin Azman.

Research question: does FinBERT news-headline sentiment improve short-term stock ranking beyond price and volume features, and does adding macro context improve it further?

The system turns 5-day forward return predictions into a weekly cross-sectional ranking of 30 US large caps (2019-2023). It does not output buy, sell or hold calls. Research project only, not financial advice.

## Status
- Phase 1 (data): complete. QA 31/31. Tag phase1-complete on main.
- Phase 2 (features): built on dev. QA gate 23/24. Open: 200-headline FinBERT label validation.
- Phase 3 (models): next.
- Branches: main = Phase 1 close (ae19c97). dev = Phase 2 work.
- Live status: PLAN.md and BACKLOG.md in this folder.

## Docs map
| File | Purpose |
|---|---|
| ../README.md | Full scope reference, single source of truth for scope |
| PLAN.md | Goals, non-goals, scope, phases |
| BACKLOG.md | Task queue: now, next, later, done |
| AGENTS.md | Rules for AI assistants working on this repo |
| DESIGN.md | Dashboard design system (Phase 6) |
| BRAND.md | Audience, voice and wording rules |
| SESSIONS.md | Running log of sessions and active state |
| ../DECISION_LOG.MD | Every design decision, dated |
| ../CONTEXT.md | Ground-truth summary for AI assistants |

## Tech stack
Python 3.12, PyTorch, Hugging Face Transformers (FinBERT), XGBoost, scikit-learn, pandas, MLflow (SQLite, local). Planned for Phase 6: PostgreSQL, FastAPI, React (local, read-only).

## Setup
```
git clone https://github.com/kedrick07/nuroquant.git
cd nuroquant
git checkout dev
python -m venv .venv
source .venv/bin/activate
pip install torch transformers huggingface_hub pyarrow pandas numpy scikit-learn xgboost mlflow python-dotenv
```
- Install the CUDA build of torch that matches your driver (pytorch.org).
- There is no requirements.txt yet (see BACKLOG.md).
- Set HUGGINGFACE_TOKEN in your shell or in a git-ignored .env file.
- data/ is git-ignored. Restore it from your backup or rebuild it with the fetch_and_validate scripts (re-run order for a new ticker list: 7, 9, 10, 11, 12, 13, 14, 16).

## Run
Phase 1 QA (expect 31/31):
```
python fetch_and_validate/16_phase1_qa.py
```
Phase 2, from the repo root, in order:
```
python features/00_snapshot.py          # data checksums and package versions
python features/01_finbert_score.py     # run once; refuses to overwrite the cache
python features/01b_finbert_sanity.py
python features/02_make_label_sample.py
python features/03_f0_technical.py
python features/04_sentiment_features.py
python features/05_macro_features.py
python features/06_targets_panel.py
python features/07_phase2_qa.py         # expect 24/24 once labels are filled
python features/08_power_sim.py
```

## Repo layout
- fetch_and_validate/: Phase 1 pipeline, scripts 1-16
- features/: Phase 2 scripts and config.py
- results/: small tracked outputs (checksums, FinBERT revision, power simulation, labels)
- data/: git-ignored, back up separately
- docs/: these documents

## Claim scope
Among 30 well-covered, 2019-era large-cap survivors over 2019-2023, does FinBERT headline sentiment improve weekly ranking beyond price features? No wider claim is made.
