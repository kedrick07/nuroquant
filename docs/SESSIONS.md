# SESSIONS

Last updated: 2026-10-26

Newest first. Each entry: what was done, findings, pending items.

## Active state (2026-10-26)
- Phase 3 is in progress. 3.0 (close-out) is done; 3.1 (CPCV splitter) is next.
- main now includes dev via PR #1 (merge commit 97729b7). Tags: phase1-complete (ae19c97), phase2-complete (c80280a). Docs commits made on dev after the merge are not yet on main.
- requirements.txt (pinned, d728b05) verified in a clean venv. data/ backed up outside git.
- Primary sentiment scorer: ProsusAI/finbert (validation macro F1 0.516). Robustness variant: yiyanghkust/finbert-tone, all articles scored.
- Environment: Fedora, Python 3.14.8 (package versions unchanged from the 2026-10-15 snapshot).
- No LSTM calendar deadline (logged 2026-10-20).

## 2026-10-26 - Phase 3.0 close-out
- Pushed requirements.txt (d728b05). Verified in a clean venv on Fedora: pip install OK, pip check found no broken requirements, torch 2.14.1+cu130 (CUDA available), transformers 5.19.0, pandas 3.0.6, xgboost 3.4.1, mlflow 3.17.1, and 07_phase2_qa.py still 24/24.
- Opened PR #1 (dev to main) and merged it with a merge commit (97729b7). The root README on main still starts with the correct first line.
- Backed up data/ outside git (confirmed by the author).
- Updated PLAN, BACKLOG and SESSIONS. Pending: README status line, CONTEXT.md and a Decision Log entry for the merge (separate commit).
- Note: requirements.txt is a full pip freeze; it could be slimmed later (rc pin on wrapt, both nvidia-nccl-cu12 and cu13 listed).

## 2026-10-20 (late) - Phase 2 close-out docs
- Pushed SETUP.md and updated SESSIONS, BACKLOG, PLAN, AGENTS (78b6a0e). The author removed the duplicate Decision Log entries (25ac6da) and merged (3a9af53); nothing was lost in the merge.
- Updated the root README (status, phase table, open decisions, validation result, scripts 09-12, 10-20 decision rows) and CONTEXT.md to the Phase 2 complete state; added a closing Decision Log entry.
- Phase progress reports are kept outside the repo (no docs/reports/).
- Merge of dev into main and tag phase2-complete: main was fast-forwarded to c80280a and tagged phase2-complete.

## 2026-10-20 - Phase 2 closed on dev
- Hand-labeled 200 headlines (101 neutral, 67 positive, 32 negative; 68 off-topic) in results/labels/headline_sample.csv; frozen copy in headline_sample_human.csv.
- FinBERT validation: macro F1 0.516 (95% bootstrap CI 0.44-0.59), accuracy 0.545, kappa 0.230. Main miss: 44 of 67 positive headlines scored neutral. The extension rule was not triggered (CI entirely below 0.65).
- Compared candidate second scorers on the labels: finbert-tone F1 0.581 (Spearman 0.564), ProsusAI 0.516 (0.449), distilroberta 0.502 (0.460). Chose yiyanghkust/finbert-tone (revision 4921590d3c0c3832c0efea24c8381ce0bda7844b) as a robustness variant; scored all 141,667 articles in 27 s. ProsusAI/finbert stays primary.
- QA check 24 now follows the written acceptance rule (F1 >= 0.65 or a second scorer queued): 24/24.
- Environment changed to Fedora, Python 3.14.8; old snapshot kept as results/phase2_snapshot_2026-10-15.json.
- Decision: no LSTM calendar deadline; if LSTM is dropped, the fallback applies and is logged before Phase 4 significance results are viewed.
- Added scripts 09-12 (labeling, F1 CI, scorer comparison, finbert-tone scoring). docs/README.md removed (root README is the README); setup and run steps moved to SETUP.md.

## 2026-10-19 to 2026-10-20 - Docs restructure
- CONTEXT.md and DECISION_LOG.MD moved into docs/ (ed01f69), paths fixed (00c8858), CONTEXT.md refreshed (8fd994d), .gitignore fixed (9762d66), root README replaced.

## 2026-10-19 - Repo review and docs setup
- Reviewed dev: two commits ahead of main (CONTEXT.md added 2026-10-14; Phase 2 scripts, results and Decision Log pushed 2026-10-16).
- Findings: labels blank; README stale; CONTEXT.md stale; .gitignore typo.
- Observation: many FNSPID headlines in the label sample are not about the tagged ticker (market wrap-ups, ETF flows, sector-update lists); the relevant column measured this.
- Created the docs/ folder: README, AGENTS, PLAN, BACKLOG, DESIGN, BRAND, SESSIONS.

## 2026-10-14 to 2026-10-15 - Phase 2 build
- Reviewed the Phase 2 plan and refined it: LSTM pre-window (features from 2019-03-04, model rows from 2019-03-29), fold-wise macro scaling, news-day-only sentiment z-scores, blind labeling protocol.
- Built and ran scripts 00-08: snapshot, FinBERT scoring (revision 0bc2ac10dda6929bcb1f77ccc43c7ec5c424f031, 141,667 texts, about 27 s on the RTX 4050), sanity checks, label sample, F0 (12 features), F1 (6), F2 raw macro (10), targets and panel, QA gate, power simulation.
- Results: fp16 vs fp32 max score diff 0.0058; NYT scores more negative than FNSPID (-0.33 vs +0.06); F0 max correlation 0.945 (no prune); panel 35,790 model rows over 1,193 dates (about 239 weekly cross-sections); target tails are real events (QCOM Apr 2019, META 2022, NEE Sep 2023); 80% power needs an IC gain of about 0.015-0.032.
- Choices logged: WTI uses a 20-day dollar change; target_ret5_xs clipped at +/-5 for the robustness run; std floor 0.10 and minimum 5 observations in sentiment z-scores.
- Decision Log: 9 entries for 2026-10-15 added.

## Earlier
- 2026-10-14: Phase 1 closed; warm-up data verified; dev merged to main; tag phase1-complete.
- 2026-10-13: evaluation design frozen; README rebuilt (see DECISION_LOG.MD).
- 2026-09-26 to 2026-10-07: scope reset, universe locked, data pipeline built.
