# AGENTS

Last updated: 2026-10-19

Rules for any AI assistant working on NuroQuant. Read this file, ../CONTEXT.md and PLAN.md before answering.

## Ground rules
1. If a fact is not in these docs or in code the user pastes, say it is not in context. Never invent file names, columns, function signatures or package APIs.
2. If the docs conflict with newer user info, trust the user and flag the conflict.
3. The design is frozen (../CONTEXT.md section 6). Any change is logged in ../DECISION_LOG.MD as a new trial before it is built.
4. Kedrick wants short, simple, practical answers. He is building his coding skill: guide and explain by default, and give full scripts when he asks.
5. While a phase is in progress, end each reply with the phase checklist and tick items as they are completed.
6. Report nulls and limits plainly. Follow BRAND.md for wording.

## Do not
- Use raw article counts as features.
- Apply log or percent transforms to WTI (negative on 2020-04-20).
- Use warm-up files for labels or evaluation.
- Use substring news matching.
- Say no survivorship bias.
- Change alignment rules (FNSPID t+1, NYT 16:00 ET) without a logged trial.
- Print tokens or commit .env. The Hugging Face token lives in the HUGGINGFACE_TOKEN environment variable.
- Commit anything under data/ (git-ignored).

## Conventions
- FIRST_MODEL_DATE (2019-03-29) and FIRST_LSTM_INPUT_DATE (2019-03-04) come from features/config.py. Never hard-code dates.
- Features at date t use data up to t only. Scalers are fit inside training folds.
- Primary target is target_rank. Primary metric is weekly rank IC.
- Every new run gets a trial_id (MLflow tag).

## Git rules
- Work on dev. Merge to main only when Kedrick says so. Tag phase ends (phaseN-complete).
- Run git pull before editing. Check the first line of README.md after any merge.
- Commit messages say what changed, for example: docs: update BACKLOG after Phase 2 QA.

## Environment (from results/phase2_snapshot.json, 2026-10-15)
Python 3.12.9, torch 2.14.1+cu130, transformers 5.19.0, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, xgboost 3.4.1, mlflow 3.17.1. GPU: RTX 4050 Laptop, 6.1 GB VRAM.

## Doc update protocol
At the end of any major change, update the docs and push them to the dev branch (docs/ folder) in one commit.

Major change = a phase milestone, a QA result, a design or scope decision, a new script or folder, a merge or tag, or a change to the dashboard design or wording rules.

| Doc | Update when |
|---|---|
| SESSIONS.md | Every major change (what was done, findings, active state) |
| BACKLOG.md | Every major change (move items, add new ones) |
| PLAN.md | Phase status, scope or risk changes |
| README.md (this folder) | Status, setup or run steps change |
| DESIGN.md, BRAND.md | UI or wording rules change |
| ../DECISION_LOG.MD, ../README.md, ../CONTEXT.md | Decisions or phase end (separate commit, ask first) |

Always show the user what will be pushed and get a yes first (the GitHub connector asks for confirmation on each write). Never push to main unless asked. Never overwrite a file from a stale copy: read the current version from GitHub first.
