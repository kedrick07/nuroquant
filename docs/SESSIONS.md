# SESSIONS

Last updated: 2026-10-19

Newest first. Each entry: what was done, findings, pending items.

## Active state (2026-10-19)
- Branch dev at ae44484 (2026-10-16). main at ae19c97 (Phase 1 close, tag phase1-complete).
- Phase 2 built. QA gate 23/24 at last run (2026-10-15). Open: 200 headline labels (blank on dev).
- Root README.md on dev is stale (32,486 bytes). The updated version (adds section 2.4.1) is not yet committed.
- CONTEXT.md still describes the 2026-10-14 state.
- LSTM fallback deadline not set. data/ backup not confirmed.

## 2026-10-19 - Repo review and docs setup
- Reviewed dev: two commits ahead of main (CONTEXT.md added 2026-10-14; Phase 2 scripts, results and Decision Log pushed 2026-10-16).
- Findings: labels blank; README stale; CONTEXT.md stale; .gitignore typo (.ipynb checkpoints should be .ipynb_checkpoints/).
- Observation: many FNSPID headlines in the label sample are not about the tagged ticker (market wrap-ups, ETF flows, sector-update lists). Plan: add a relevant column while labeling to measure the off-topic rate.
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
