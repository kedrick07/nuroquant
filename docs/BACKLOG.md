# BACKLOG

Last updated: 2026-10-19

Order = priority. Move items between sections; never delete done items.

## Now
- [ ] Label the 200 headlines in results/labels/headline_sample.csv (blind to model scores; off-topic headlines = neutral; add a relevant column, 1 or 0).
- [ ] Run python features/07_phase2_qa.py and confirm 24/24 (macro F1 >= ~0.65, else queue a second scorer).
- [x] Commit the updated root README.md (adds section 2.4.1; stale on dev) and refresh CONTEXT.md for Phase 2.
- [ ] Back up data/ outside git (checksums are in results/phase2_snapshot.json).
- [ ] Set the LSTM fallback deadline date (end of Phase 3) and log it.
- [ ] Log the FinBERT validation result in DECISION_LOG.MD.
- [ ] Merge dev into main and tag phase2-complete.
- [ ] Write the Phase 2 progress report.

## Next (Phase 3)
- [ ] Add requirements.txt with pinned versions (see AGENTS.md environment).
- [ ] Fold-wise transformer: macro z-score on training folds, clip |z| at 5, build the 8 interactions (results/macro_recipe.json).
- [ ] CPCV splitter (N=8, k=2, purge 5, embargo 20) with a unit test for label-window overlap.
- [ ] MLflow setup: SQLite backend, tags trial_id, trial_family, counts_toward_N; log git hash and data checksums.
- [ ] Run order: Ridge, XGBoost, Random Forest, LSTM; same splits and seeds for all.
- [ ] Nested tuning (inner 4-fold purged blocks, LSTM 2), small grids, 3 seeds.
- [ ] XGBoost robustness runs: mask NVDA gap, exclude META, target_ret5_xs clipped at +/-5, 60-day embargo, summary_extra.
- [ ] Milestone CSV export of MLflow runs to results/.

## Later
- [ ] Phase 4: significance tests (block bootstrap, HAC DM with HLN, BH and BY), group SHAP and permutation importance, static GICS sector map (groups >= 3), Ken French factor data check for 2019-2023.
- [ ] Phase 5: weekly long-short backtest per CPCV path, cost grid, 1-day delay, exposure diagnostics, deflated Sharpe.
- [ ] Phase 6: read-only dashboard (4 pages) per DESIGN.md, capped at about 10% of time.
- [ ] TFT stretch run if time allows.
- [ ] Thesis chapters, poster and presentation from frozen results.

## Done
- [x] Phase 1: universe locked (30 tickers), prices, news, macro, QA 31/31, warm-up data (2026-10-07 to 2026-10-14).
- [x] Evaluation design frozen and pre-registered (2026-10-09 to 2026-10-13).
- [x] Tag phase1-complete on main (ae19c97).
- [x] FinBERT run once, revision pinned, fp16 vs fp32 check passed (2026-10-15).
- [x] F0, F1 and F2 features, targets and panel built (35,790 model rows) (2026-10-15).
- [x] Power simulation saved (2026-10-15).
- [x] Phase 2 scripts 00-08, results and Decision Log entries pushed to dev (2026-10-16).
- [x] docs/ folder created (2026-10-19).
- [x] Docs restructure and path fixes (2026-10-19).
