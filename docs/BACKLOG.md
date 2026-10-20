# BACKLOG

Last updated: 2026-10-20

Order = priority. Move items between sections; never delete done items.

## Now
- [ ] Back up data/ outside git (checksums are in results/phase2_snapshot.json).
- [ ] Remove duplicate 2026-10-20 entries in DECISION_LOG.MD (second scorer and environment each appear twice).
- [ ] Refresh CONTEXT.md for the Phase 2 close (QA 24/24, second scorer, environment).
- [ ] Merge dev into main and tag phase2-complete (git tag phase2-complete; git push --tags).
- [ ] Write the Phase 2 progress report; add the Phase 1 report and the report template to docs/reports/.

## Next (Phase 3)
- [ ] Add requirements.txt with pinned versions (see AGENTS.md environment).
- [ ] Fold-wise transformer: macro z-score on training folds, clip |z| at 5, build the 8 interactions (results/macro_recipe.json).
- [ ] CPCV splitter (N=8, k=2, purge 5, embargo 20) with a unit test for label-window overlap.
- [ ] MLflow setup: SQLite backend, tags trial_id, trial_family, counts_toward_N; log git hash and data checksums.
- [ ] Run order: Ridge, XGBoost, Random Forest, LSTM; same splits and seeds for all.
- [ ] Nested tuning (inner 4-fold purged blocks, LSTM 2), small grids, 3 seeds.
- [ ] Alt-sentiment features from finbert_tone_scores.parquet: parameterize 04_sentiment_features.py (scores file, output suffix) so the primary F1 files are never overwritten.
- [ ] XGBoost robustness runs: mask NVDA gap, exclude META, target_ret5_xs clipped at +/-5, 60-day embargo, summary_extra, finbert-tone sentiment.
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
- [x] docs/ folder created; docs restructure and path fixes (2026-10-19).
- [x] Root README and CONTEXT.md refreshed for Phase 2 (2026-10-20).
- [x] 200 headlines hand-labeled and committed (2026-10-20).
- [x] FinBERT validation: macro F1 0.516 (CI 0.44-0.59), below 0.65; logged (2026-10-20).
- [x] Second scorer chosen (finbert-tone, F1 0.581) and run on all articles (2026-10-20).
- [x] QA gate acceptance rule implemented; Phase 2 QA 24/24 (2026-10-20).
- [x] Environment change and snapshot re-recorded; LSTM fallback decision logged (2026-10-20).
- [x] Scripts 09-12 pushed to dev (2026-10-20).
