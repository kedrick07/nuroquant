2026-09-26 | Dropped NS1/NS2/NS3 news-source ablation; merged to a single FNSPID + NYT source.

2026-09-26 | Stock universe reduced from 50 to 30. Timeline changed from 2016–2026 to 2019–2023.

2026-09-26 | Ablation matrix restructured into feature-tier × model-family axes. Phases 6–7 scoped down to a local-only dashboard; cloud deployment cut. TFT (E6) confirmed as stretch goal, not required for the core conclusion. No sector cap: coverage gate + market-cap ranking decide the final mix.

2026-09-27 | Market-cap ranking switched to batched download × shares-outstanding proxy; 26 delisted/renamed tickers left unranked.

2026-09-28 | Macro set expanded from 5 to 9 FRED series; BAA10Y replaces the BAML high-yield spread (its FRED history is a rolling ~3-year window).

2026-09-29 | Daily OHLCV confirmed as price granularity. FinBERT text: FNSPID title only, NYT headline + abstract. Macro: CPI converted to YoY; CPI/UNRATE/FEDFUNDS shifted to approximate release dates. NYT matching rewritten (word-boundary, case-sensitive, tiered) after substring false positives.

2026-10-02 | Cleaning rules fixed (tag-only NYT matches dropped, news sections only, dedupe). Trading-day alignment fixed (FNSPID t+1, NYT 16:00 ET). Coverage gate re-run on cleaned data: CSCO, JPM, NVDA failed; MSFT, NFLX, DIS <100% months.

2026-10-07 | Strict 100%-month rule applied (NVDA documented exception): CSCO, JPM, MSFT, NFLX, DIS replaced by CMCSA, TXN, PYPL, WFC, QCOM; pipeline re-run. Phase 1 QA passed 31/31; Phase 1 closed.

2026-10-09 | Backtest benchmark changed from SPY to equal-weight 30-stock universe; F0 model, random ranking and a simple reversal/momentum rule added as baselines. SPY dropped because the universe is not representative of the index.

2026-10-09 | Pre-registered hypotheses H1 (F1>F0) and H2 (F2>F1, XGBoost). Decision rule: paired weekly rank-IC difference must have a BH-adjusted two-sided p < 0.05 AND a block-bootstrap 95% CI excluding zero; H1 requires this in ≥3 of 4 families. Null results reported as findings.

2026-10-09 | Significance: weekly non-overlapping rank IC; paired difference tested via block bootstrap (primary) and HAC/DM with HLN correction (cross-check); BH correction over 5 primary tests; power analysis before modeling.

2026-10-09 | Targets: target_ret5 = adj close t→t+5 trading days (evaluation/P&L only); target_rank = centered cross-sectional percentile rank (primary training target, all models); target_ret5_xs = demeaned, vol-scaled return (XGBoost robustness only). Backtest tests 1-day execution delay.

2026-10-09 | Features: technical features converted to stationary ratios/returns; technical and sentiment features cross-sectionally rank-normalized per date; macro z-scored over time on training folds only; prices fetched from 2018-07 for rolling-window warm-up (separate file, see 2026-10-13 warm-up entry); identical normalized inputs for all models.

2026-10-09 | Sentiment: 6 daily features (sent_mean, sent_ewm5, sent_mom, abn_count, has_news, days_since_news); per-ticker 60d z-score then cross-sectional rank; no synthetic imputation; FNSPID t+1 kept; robustness runs without META and without NVDA gap months.

2026-10-09 | FinBERT: ProsusAI/finbert (pinned revision), max_length 160, fp16, run once and cached; no fine-tuning; 200-headline stratified hand-label validation (macro F1 ≥ ~0.65 target, Cohen's kappa if two labelers); second scorer only as robustness if validation fails.

2026-10-11 | Macro (F2): stationary signals plus 8 pre-specified macro × {beta, vol} interactions; beta and vol added to F0; no sector dummies; H2 flagged as exploratory/likely null.

2026-10-13 | CPCV: N=8 date-based groups, k=2 (28 splits, 7 paths); purge 5 trading days (label horizon); embargo 20 trading days (≥ LSTM lookback), 60-day embargo sensitivity on XGBoost; same splits/seeds for all runs; per-date IC averaged across paths before testing; unit test for label-window overlap.

2026-10-13 | No separate holdout (power: ~250 weekly cross-sections in total; 2023 holds ~33% of FNSPID volume; CPCV gives 7 paths). Safeguards: frozen design, no post-hoc changes without logging as new trial, all trials tracked for deflated Sharpe, yearly breakdown exploratory. Nested tuning: inner 4-fold purged blocks (LSTM 2), metric = weekly rank IC, identical search budget per tier, small regularized grids, 3 seeds averaged.

2026-10-13 | Metrics: primary = weekly rank IC (decides H1/H2); secondary = ICIR, IC hit rate, quintile returns/spread, OOS R² on target_rank, net Sharpe (bootstrap CI), max drawdown, turnover, break-even cost, beta/alpha vs equal-weight; directional accuracy dropped (pairwise accuracy optional); group-level SHAP + group permutation importance.

2026-10-13 | Portfolio: weekly (5-day) rebalance, 5 staggered start offsets averaged; main = dollar-neutral long-short top/bottom quintile (6/6, equal weight); secondary = long-only top quintile vs equal-weight 30 and rank-weighted dollar-neutral; entry at close t, 1-day-delay sensitivity; backtested per CPCV path (7), median + range reported.

2026-10-13 | Costs: charged on traded notional only; base 5 bps/side + 0.5% annual borrow on shorts; sensitivity grid 0/5/10/20 bps and 0.25/0.5/1/3% borrow; Feb–Apr 2020 3× stress; break-even cost reported (gross return ÷ turnover); optional entry/exit buffer variant.

2026-10-13 | Exposure diagnostics (exploratory, outside BH family): Newey-West regression on equal-weight-30 return; Fama-French + MOM + ST-reversal regression; weekly net sector exposure and sector-neutral IC (static GICS mapping, groups ≥3 stocks); top-3 P&L concentration; beta-neutral variant if |β|>0.15 or significant.

2026-10-13 | Multiple testing: primary family = 5 IC tests (H1×4 families, H2), BH q=0.05 on two-sided bootstrap p-values, BY as conservative check; H1 passes if ≥3/4 BH-adjusted p<0.05; all other tests = separate exploratory BH family. Deflated Sharpe: weekly net SR, nominal N counted from tracker (core + robustness + variant runs; cost grids and nested tuning excluded), effective-N sensitivity, bar DSR ≥ 0.95; trial_id logged for every run.

2026-10-13 | Experiment tracking: MLflow only (SQLite backend, local artifacts); W&B dropped. Parent run per model×tier×variant, child run per CPCV split; tags trial_id/trial_family/counts_toward_N feed deflated Sharpe N; data checksums + git hash logged; runs never deleted; milestone CSV export committed to a tracked folder (e.g. results/), not under the git-ignored data/.

2026-10-13 | Dashboard: read-only viewer of frozen precomputed results (Postgres → FastAPI → React, local); 4 pages (Results, Backtest, Rankings, Explainability/Exposures); no inference/auth/live data; run IDs + data checksum shown; limitations banner; built after Phase 5 freeze, capped ~10% time, first item cut if behind schedule.

2026-10-13 | Deep models: LSTM spec frozen (20d lookback, date-batched, 3 seeds, ~1 GPU-day cap); time-based fallback (if LSTM F0/F1 not done by end of Phase 3, H1 rule becomes ≥2 of 3 families, BH over 4 tests); TFT stretch outside primary family. Robustness (XGBoost F0/F1 only): NVDA gap masked, META excluded, target_ret5_xs, 60d embargo, summary_extra; consistency = same sign + overlapping CI; exploratory family.

2026-10-13 | Limitations documented: survivorship (458 survivors), market-cap proxy look-ahead, FNSPID-availability selection (13 large caps excluded), FNSPID = 98.2% of news (NYT 1.8%), uneven day coverage, sector skew, mega-cap-only validity, short regime-specific sample (power), t+1 conservative timing, vendor/vintage/FinBERT-domain caveats, static GICS snapshot, simplified trading assumptions. Claims scoped to this universe and period.

2026-10-13 | Warm-up data: 2018-07 prices (separate file ohlcv_warmup.csv) and untrimmed 2017+ macro are used only to compute rolling features, never for labels or evaluation. News stays 2019–2023 (no 2018 news exists in FNSPID/NYT pulls). Phase 1 QA window check on ohlcv_daily.csv unchanged.

2026-10-13 | First ~60 trading days of 2019 dropped from ALL tiers (F0/F1/F2) because trailing-60-day sentiment features are not fully formed there; this keeps paired F1-vs-F0 tests on identical dates. Implemented as a single FIRST_MODEL_DATE (61st trading day of 2019), not a hard-coded calendar date.

2026-10-14 | Warm-up data fetched and verified (ohlcv_warmup.csv: 30 tickers × 126 days; fred_daily_full.csv: 2018-07-01 → 2023-12-31, 0 NaN); boundary join checked (max |move| 3.91%); Phase 1 QA re-run 31/31; Phase 1 closed.