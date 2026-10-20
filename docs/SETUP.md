# SETUP

Last updated: 2026-10-20

Setup and run steps for NuroQuant. The root README.md is the scope reference; this file replaces the old docs/README.md.

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
| DECISION_LOG.MD | Every design decision, dated |
| CONTEXT.md | Ground-truth summary for AI assistants |

## Setup
```
git clone https://github.com/kedrick07/nuroquant.git
cd nuroquant
git checkout dev
python -m venv .venv
source .venv/bin/activate
pip install torch transformers huggingface_hub pyarrow pandas numpy scikit-learn scipy xgboost mlflow python-dotenv
```
- Install the CUDA build of torch that matches your driver (pytorch.org).
- There is no requirements.txt yet (see BACKLOG.md). Versions in use are in ../results/phase2_snapshot.json (Python 3.14.8; the original Python 3.12.9 snapshot is ../results/phase2_snapshot_2026-10-15.json).
- Set HUGGINGFACE_TOKEN in your shell or in a git-ignored .env file. Never commit it.
- data/ is git-ignored. Restore it from your backup or rebuild it with the fetch_and_validate scripts (re-run order for a new ticker list: 7, 9, 10, 11, 12, 13, 14, 16).

## Run
Phase 1 QA (expect 31/31):
```
python fetch_and_validate/16_phase1_qa.py
```
Phase 2, from the repo root, in order:
```
python features/00_snapshot.py            # data checksums and package versions
python features/01_finbert_score.py       # run once; refuses to overwrite the cache
python features/01b_finbert_sanity.py
python features/02_make_label_sample.py   # do NOT re-run after labeling: it overwrites the labels file
python features/03_f0_technical.py
python features/04_sentiment_features.py
python features/05_macro_features.py
python features/06_targets_panel.py
python features/07_phase2_qa.py           # expect 24/24
python features/08_power_sim.py
python features/09_label_headlines.py     # interactive hand labeling, resumable
python features/10_finbert_f1_ci.py       # macro F1 with bootstrap CI
python features/11_compare_scorers.py     # candidate second scorers vs the hand labels
python features/12_score_finbert_tone.py  # second scorer, run once
```

## Repo layout
- fetch_and_validate/: Phase 1 pipeline, scripts 1-16
- features/: Phase 2 scripts and config.py
- results/: small tracked outputs (checksums, scorer revisions, power simulation, labels)
- data/: git-ignored, back up separately
- docs/: project documents
