# NuroQuant README Update Patch (2026-10-13)

Paste these sections into README.md. Replace the sections named in each heading. Nothing here changes Phase 1 data work.

---

## Status line (replace the top status line)

**Status (2026-10-13): Phase 1 complete (QA 31/31). Evaluation design frozen (see 2.6 and Decision Log). Next: Phase 2 (feature engineering).**

---

## 1.1 Hypotheses and decision rules (NEW, insert after Section 1)

- **H1 (sentiment):** F1 has higher weekly rank IC than F0.
- **H2 (macro):** F2 has higher weekly rank IC than F1 (XGBoost only). Flagged as the weaker hypothesis: a null result is plausible and will be reported as a finding.
- **Primary metric:** mean weekly rank IC on non-overlapping 5-day cross-sections, out of sample.
- **H1 supported if:** the paired weekly IC difference (F1 - F0) is significant (BH-adjusted two-sided p < 0.05, bootstrap 95% CI excluding zero) in at least 3 of 4 model families.
- **H2 supported if:** the same test passes for XGBoost F2 vs F1.
- **Fallback (time-based, not performance-based):** if the LSTM F0/F1 runs are not complete by the end of Phase 3, drop LSTM from the primary family; H1 then needs 2 of 3 families and BH runs over 4 tests.
- Secondary evidence (net Sharpe, SHAP shares, exposures) supports the story but never decides H1/H2.

---

## 2.1.3 Known Limitations (NEW, add after 2.1.2)

1. Survivorship: 458 of 500 Jan-2019 constituents survive the continuous-listing filter. Shorts look less profitable and longs more profitable than in a full universe. Do not claim "no survivorship bias".
2. Market-cap screen uses the 2019-2023 average close x current shares (look-ahead in the screen; mild tilt toward growers). It only orders the screen.
3. Selection by news availability: BRK.B, JNJ, UNH, MA, PG, HD, LLY, AVGO, BAC, PFE, VZ, MCD (plus LIN) screened out for thin FNSPID coverage.
4. News is effectively one source: FNSPID 98.2% (139,075), NYT 1.8% (2,592). Sentiment = FNSPID headline sentiment in practice (META is the exception, 63% NYT).
5. Uneven coverage: 18.6% of ticker-days have no news; low day coverage for META (42.1%), AMZN (49.2%), AAPL (52.7%); NVDA has 11 gap months; FNSPID volume rises by year (2019: 21,045; 2023: 46,489).
6. Sector skew: ~10 tech names, 1 bank, 1 utility, no industrials/materials/real estate.
7. Efficient mega caps only: a null result does not say sentiment is useless in smaller stocks.
8. Short regime-specific sample: ~250 weekly cross-sections of 30 stocks; minimum detectable IC difference roughly 0.02-0.034 (to be confirmed by power simulation).
9. FNSPID timestamps are date-only, assigned t+1 (conservative: understates sentiment value).
10. Prices from yfinance (adjusted); macro series as-currently-published with approximate release lags; FinBERT trained on financial sentences, not headlines (validated on 200 hand-labeled headlines).
11. Sector labels use one static GICS snapshot (V and PYPL were reclassified in 2023).
12. Simplified trading assumptions: flat costs, flat borrow rate, no market impact, shorting always available.

**Claim scope:** among 30 well-covered, 2019-era large-cap survivors over 2019-2023, does FinBERT headline sentiment improve weekly ranking beyond price features?

---

## 2.4 Feature Set (REPLACE)

- **Technical (all stationary, scale-free):** returns over 1/5/10/20/60 trading days; close/SMA20 - 1; close/SMA50 - 1; MACD/close; RSI; 20-day realized volatility; log(volume / 20-day average volume); rolling 60-day beta to the equal-weight universe return. No raw price or volume levels.
- **Sentiment (6):** sent_mean, sent_ewm5 (half-life ~3 days), sent_mom (3-day ewm minus 20-day ewm), abn_count (log(1+n) z-scored vs the ticker's trailing 60 days), has_news, days_since_news (capped at 10). Article score = p_pos - p_neg. No synthetic imputation: no-news days keep score 0 and are flagged.
- **Macro (stationary):** 5-day change in VIX; 5-day change in 10Y; curve slope (10Y - 2Y) and its change; 20-day change in BAA10Y; 20-day return of WTI and of the USD index; CPI, unemployment and fed funds changes. Z-scored over time on training folds only.
- **Macro x stock interactions (8, pre-specified):** {dVIX, d10Y, dBAA10Y, WTI 20d return} x {rolling beta, 20d volatility}. No sector dummies.
- **Tiers:** F0 = technical (incl. beta, vol); F1 = F0 + sentiment; F2 = F1 + macro + 8 interactions.
- **Normalization:** technical and sentiment features rank-normalized per date across the 30 stocks (centered, about -0.5 to +0.5); binary flags unchanged; sentiment score features first z-scored per ticker over a trailing 60 days. Macro is not cross-sectionally ranked. Rolling windows use past data only; prices fetched from 2018-07 for warm-up. Scalers are fit inside training folds. Identical normalized inputs for every model.
- **Targets:** `target_ret5` = adj close t -> t+5 trading days (evaluation and P&L only). `target_rank` = per-date centered percentile rank, (rank - 0.5)/30 - 0.5 (primary training target, all models). `target_ret5_xs` = cross-sectionally demeaned, volatility-scaled return (XGBoost robustness only). Last 5 days of 2023 and dates with fewer than 30 valid labels are dropped.
- **Sentiment model:** ProsusAI/finbert (pinned revision), max_length 160, fp16, run once and cached; no fine-tuning; validated on 200 hand-labeled headlines (target macro F1 about 0.65; Cohen's kappa if two labelers); a second scorer only as a robustness run if validation fails.

---

## 2.6 Evaluation (REPLACE)

**Cross-validation**
- CPCV, date-based groups: N = 8, k = 2 (28 splits, 7 paths). Purge 5 trading days (label horizon); embargo 20 trading days (>= LSTM lookback); 60-day embargo sensitivity on XGBoost. Same splits and seeds for every model and tier; unit test that no training label window overlaps a test date.
- No separate holdout (power; 2023 holds about 33% of FNSPID volume). Safeguards: frozen design, any post-hoc run logged as a new trial, yearly results exploratory.
- Nested tuning: inner 4-fold purged blocks (LSTM: 2), metric = weekly rank IC, identical search budget per tier, small regularized grids (see 2.5/Decision Log), 3 seeds averaged for stochastic models.

**Significance**
- Unit: weekly non-overlapping rank IC. Paired difference d_t tested with a block bootstrap (primary, 10,000 resamples) and HAC Diebold-Mariano with HLN correction (cross-check). Per-date IC averaged across the 7 paths before testing; paths are not treated as independent samples.
- Primary family: 5 tests (H1 x 4 families, H2), BH at q = 0.05, BY as conservative check. All other tests: separate exploratory BH family.
- Power simulation before modeling.
- Deflated Sharpe ratio on weekly net Sharpe; N counted from tracker tags (core + robustness + variants; cost grids and nested tuning excluded); effective-N sensitivity; bar DSR >= 0.95.

**Metrics**
- Primary: weekly rank IC. Secondary: ICIR, IC hit rate, quintile returns and spread, OOS R2 on target_rank, net Sharpe (bootstrap CI), max drawdown, Calmar, turnover, break-even cost, beta/alpha vs equal-weight. Directional accuracy dropped (pairwise accuracy optional).
- Interpretability: group-level SHAP shares plus group permutation importance.

**Backtest**
- Weekly (5-day) rebalance, 5 staggered start offsets averaged. Main: dollar-neutral long-short, top/bottom quintile (6/6, equal weight). Secondary: long-only top quintile vs equal-weight 30; rank-weighted dollar-neutral. Entry at close t; 1-day-delay sensitivity. Run per CPCV path (7); report median and range.
- Benchmarks (SPY dropped): equal-weight 30-stock universe (primary), F0 model, random ranking (many seeds), 5-day reversal rule.
- Costs: charged on traded notional; base 5 bps per side + 0.5% annual borrow on shorts; grid 0/5/10/20 bps and 0.25/0.5/1/3% borrow; Feb-Apr 2020 3x stress; break-even cost reported; optional entry/exit buffer variant (enter top 20%, exit below 40%).
- Exposure diagnostics (exploratory): Newey-West regression on equal-weight-30 return; Fama-French + momentum + short-term reversal regression; weekly net sector exposure and sector-neutral IC (static GICS, groups >= 3 stocks); top-3 P&L concentration; beta-neutral variant if |beta| > 0.15 or significant.

**Robustness runs (XGBoost F0/F1 only, exploratory family):** NVDA gap months masked; META excluded; target_ret5_xs; 60-day embargo; FNSPID summary_extra added (F1 only). Consistent = same sign of F1-F0 and overlapping CI.

---

## 2.7 Technology Stack (REPLACE experiment tracking and serving lines)

- **Experiment tracking:** MLflow only (SQLite backend, local artifacts). Parent run per model x tier x variant; child run per CPCV split. Tags: trial_id, trial_family, counts_toward_N. Log git hash and data checksums. Never delete runs. Commit a milestone CSV export. W&B removed.
- **Serving:** read-only viewer of frozen precomputed results: PostgreSQL -> FastAPI -> React, local only. Four pages: Results, Backtest, Rankings, Explainability/Exposures. Built after Phase 5 freeze, capped at about 10% of project time, first item cut if behind schedule. Write the FastAPI endpoints and schema yourself; Lovable for React scaffolding only.
- **LSTM spec:** 20-day lookback, date-batched, hidden {32, 64}, 1-2 layers, dropout {0.2, 0.4}, AdamW, early stopping (patience 5, max 50 epochs), 3 seeds, about 1 GPU-day cap.
- **Run order:** Ridge -> XGBoost -> Random Forest -> LSTM -> robustness -> TFT (stretch, outside primary family).

---

## 3. Phase Plan (REPLACE rows 5 and 6)

| Phase | Focus | Status |
|---|---|---|
| 5 | Backtesting (long-short vs equal-weight 30, F0, random, reversal baselines) | Planned |
| 6 | Read-only results dashboard (FastAPI + React, local) | Planned, built last, first to cut |

---

## 4. Open Decisions (REPLACE)

- [ ] Static sector mapping for the 30 tickers (groups >= 3 stocks) and snapshot date
- [ ] Hand-label the 200-headline FinBERT validation sample (and decide second labeler)
- [ ] Finalize exact technical feature list and check correlations (drop > 0.95)
- [ ] Power simulation to confirm the minimum detectable IC difference
- [ ] Set the end-of-Phase-3 deadline date for the LSTM fallback rule
- [ ] Confirm Ken French factor data covers 2019-2023 and aligns to the trading calendar
- [ ] Re-fetch prices from 2018-07 for rolling-window warm-up

Resolved: final 30 tickers; benchmark; target construction; CPCV configuration; NVDA robustness design; tracker; dashboard scope.

---

## 6. Decision Log (APPEND)

| Date | Change |
|---|---|
| 2026-10-09 | Backtest benchmark changed from SPY to equal-weight 30-stock universe; F0 model, random ranking and 5-day reversal rule added as baselines. SPY dropped because the universe is not representative of the index |
| 2026-10-09 | Pre-registered hypotheses H1 (F1>F0) and H2 (F2>F1, XGBoost); H1 requires >= 3 of 4 families; null results reported as findings |
| 2026-10-09 | Significance: weekly non-overlapping rank IC; paired difference via block bootstrap (primary) and HAC/DM with HLN (cross-check); BH over 5 primary tests; power analysis before modeling |
| 2026-10-09 | Targets: target_ret5 (evaluation/P&L), target_rank (primary training target), target_ret5_xs (XGBoost robustness); backtest tests 1-day execution delay |
| 2026-10-09 | Features: stationary technical ratios; per-date cross-sectional rank; macro z-scored on training folds only; prices from 2018-07 for warm-up; identical inputs for all models |
| 2026-10-11 | Macro (F2): stationary signals plus 8 pre-specified macro x {beta, vol} interactions; beta and vol added to F0; no sector dummies; H2 flagged as exploratory/likely null |
| 2026-10-11 | Sentiment: 6 daily features; per-ticker 60d z-score then cross-sectional rank; no synthetic imputation; FNSPID t+1 kept; robustness without META and without NVDA gap months |
| 2026-10-11 | FinBERT: ProsusAI/finbert pinned, max_length 160, fp16, run once, no fine-tuning; 200-headline hand-label validation; second scorer only as robustness |
| 2026-10-13 | CPCV: N=8, k=2 (28 splits, 7 paths); purge 5, embargo 20 trading days; 60-day embargo sensitivity; per-date IC averaged across paths |
| 2026-10-13 | No separate holdout (power; 2023 holds ~33% of FNSPID volume). Safeguards: frozen design, post-hoc changes logged as new trials. Nested tuning: inner 4-fold purged blocks (LSTM 2), metric = weekly rank IC |
| 2026-10-13 | Metrics: primary = weekly rank IC; secondary = ICIR, hit rate, quintile spread, OOS R2, net Sharpe, drawdown, turnover, break-even cost, beta/alpha; directional accuracy dropped |
| 2026-10-13 | Portfolio: weekly rebalance, 5 staggered offsets; dollar-neutral top/bottom quintile (6/6); secondary long-only top quintile and rank-weighted variants; entry at close t; per-path backtests |
| 2026-10-13 | Costs: base 5 bps/side + 0.5% borrow; grid 0/5/10/20 bps and 0.25/0.5/1/3%; Feb-Apr 2020 3x stress; break-even cost; optional buffer variant |
| 2026-10-13 | Exposure diagnostics added (market, Fama-French + MOM + ST reversal, sector, concentration); beta-neutral variant if |beta| > 0.15 or significant |
| 2026-10-13 | Multiple testing: BH q=0.05 on 5 primary IC tests, BY check, exploratory family separate; deflated Sharpe with tracker-counted N, bar DSR >= 0.95 |
| 2026-10-13 | Experiment tracking: MLflow only; W&B dropped |
| 2026-10-13 | Dashboard: read-only frozen results viewer, 4 pages, local, built after Phase 5, ~10% time cap |
| 2026-10-13 | LSTM spec frozen with ~1 GPU-day cap and time-based fallback (H1 becomes >= 2 of 3 if not done by end of Phase 3); TFT stretch outside primary family; XGBoost-only robustness list |
| 2026-10-13 | Limitations documented (2.1.3); claims scoped to this universe and period |
