# BACKLOG

Last updated: 2026-10-27

Order = priority. Move items between sections; never delete done items.

## Now (3.3)
- [ ] 3.3 MLflow setup: SQLite backend, tags trial_id, trial_family, counts_toward_N; log git hash and data checksums; fix seeds and splits. Parent run per model x tier x variant, child run per CPCV split.

## Next (Phase 3)
- [ ] 3.4 Run order: Ridge, XGBoost, Random Forest; same splits and seeds for all.
- [ ] 3.4 Nested tuning (inner 4-fold purged blocks, LSTM 2), small grids, 3 seeds.
- [ ] 3.5 LSTM F0/F1 (20-day lookback, no window needs rows before 2019-03-04).
- [ ] 3.6 Alt-sentiment features from finbert_tone_scores.parquet: parameterize 04_sentiment_features.py (scores file, output suffix) so the primary F1 files are never overwritten.
- [ ] 3.6 XGBoost robustness runs: mask NVDA gap, exclude META, target_ret5_xs clipped at +/-5, 60-day embargo, summary_extra, finbert-tone sentiment.
- [ ] 3.7 Milestone CSV export of MLflow runs to results/; Phase 3 QA gate and report.
- [ ] Optional: slim requirements.txt (it is a full pip freeze; check wrapt rc pin and the duplicate nvidia-nccl entries).
- [ ] Decision Log entry for the 3.2 implementation details (separate commit, author appends locally).
- [ ] 3.4 note: the inner 4-fold purged blocks for nested tuning need their own purge and embargo; reuse the same rule as models/cpcv.py.
- [ ] Phase 3 report note: d_unrate_21 clips on about 125 dates when 2020 is held out (21 when it is in training); mention when interpreting H2.

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
- [x] Duplicate Decision Log entries removed (2026-10-20).
- [x] Root README and CONTEXT.md updated to the Phase 2 complete state; closing Decision Log entry added (2026-10-20).
- [x] Phase progress reports are kept outside the repo, not in docs/reports/ (decided 2026-10-20).
- [x] Phase 2 closed on dev (2026-10-20).
- [x] Tag phase2-complete on main (c80280a) (2026-10-20).
- [x] requirements.txt with pinned versions added (d728b05, 2026-10-26).
- [x] requirements.txt verified in a clean venv: pip install OK, pip check clean, versions match the snapshot, 07_phase2_qa.py 24/24 (2026-10-26).
- [x] data/ backed up outside git (confirmed by the author, 2026-10-26).
- [x] dev merged into main via PR #1, merge commit 97729b7 (2026-10-26).
- [x] 3.0 docs synced: PLAN, BACKLOG, SESSIONS (f7ffb4a), README and Decision Log (85ac10a), CONTEXT.md (8b0bb84) (2026-10-26).
- [x] 3.1 CPCV splitter: models/cpcv.py (8 contiguous groups, 28 splits, purge 5 before, embargo 20 after, 7 paths) and tests/test_cpcv.py (8 tests pass, incl. two mutation tests); pytest 9.1.1 pinned (7676aac, 2026-10-27).
- [x] 3.1 docs and Decision Log entry (001cfbe, e081814) (2026-10-27).
- [x] 3.2 Fold-wise macro transformer: models/transform.py (FoldMacroTransformer: fit on training dates, ddof=0, clip +/-5, 8 interactions) and tests/test_transform.py (7 tests pass, 15 total) (b3334ba, 2026-10-27).
