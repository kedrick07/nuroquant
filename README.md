# NuroQuant — Project Scope & Reference Document

**Author:** Kedrick Selvanesan · Dept. of Multimedia, FSKTM, UPM
**Supervisor:** Assoc. Prof. Dr. Azreen bin Azman
**Type:** Final Year Project

This document is the single source of truth for project scope. When a decision changes, update it here first — everything else (code, thesis chapters, Gantt chart) should follow this file, not the other way around.

**Status (2026-10-13): Phase 1 complete (QA 31/31). Evaluation design frozen (see 2.6 and Decision Log). Next: Phase 2 (feature engineering).**

---

## 1. Core Research Question

> Does news sentiment (FinBERT-derived) measurably improve short-term equity ranking performance beyond price/volume features alone — and does adding macroeconomic context improve it further?

Secondary framing: NuroQuant predicts continuous 5-day forward returns and converts them into a **cross-sectional ranking** of a large-cap stock universe, rather than issuing buy/sell/hold classifications.

### 1.1 Hypotheses and Decision Rules

- **H1 (sentiment):** F1 has higher weekly rank IC than F0.
- **H2 (macro):** F2 has higher weekly rank IC than F1 (XGBoost only). Flagged as the weaker hypothesis: a null result is plausible and will be reported as a finding.
- **Primary metric:** mean weekly rank IC on non-overlapping 5-day cross-sections, out of sample.
- **H1 supported if:** the paired weekly IC difference (F1 - F0) is significant (BH-adjusted two-sided p < 0.05, bootstrap 95% CI excluding zero) in at least 3 of 4 model families.
- **H2 supported if:** the same test passes for XGBoost F2 vs F1.
- **Fallback (time-based, not performance-based):** if the LSTM F0/F1 runs are not complete by the end of Phase 3, drop LSTM from the primary family; H1 then needs 2 of 3 families and BH runs over 4 tests.
- Secondary evidence (net Sharpe, SHAP shares, exposures) supports the story but never decides H1/H2.

---

## 2. Finalized Scope

### 2.1 Stock Universe
- **Size:** 30 stocks (revised down from 50)
- **Selection method:**
  1. S&P 500 constituents as of **Jan 2019**
  2. Keep tickers continuously listed through **2023** (no delistings, mid-window IPOs, or major spin-offs/mergers) → 458 survivors
  3. Rank survivors by average market cap over the window (proxy: mean daily close 2019–2023 × current shares outstanding; used only to order the screen)
  4. **Coverage gate**, applied to the *cleaned* news data (FNSPID + NYT): average ≥5 articles/month AND ≥85% of the 60 months with ≥1 article
  5. **Reliability filter** (strict): keep only tickers with 100% month coverage, replacing failures with the next-ranked ticker that has 100% FNSPID month coverage. **One documented exception: NVDA** (see 2.1.2)
  6. Same-company duplicate share classes excluded (GOOGL; GOOG kept)
- **Status:** Locked (2026-10-07). See 2.1.1.

#### 2.1.1 Final 30 Tickers

Coverage numbers are from the cleaned, trading-day-aligned news table (`news_coverage_report.csv`). "Days with news" is the share of the 1,258 trading days with ≥1 article; the rest are flagged as no-news (score 0).

| # | Mkt-cap rank | Ticker | Company | Avg articles/mo | Months covered | Days with news |
|---|---|---|---|---|---|---|
| 1 | 1 | AAPL | Apple | 158.4 | 100% | 52.7% |
| 2 | 3 | AMZN | Amazon | 93.3 | 100% | 49.2% |
| 3 | 4 | GOOG | Alphabet (Class C) | 171.1 | 100% | 98.7% |
| 4 | 6 | META | Meta Platforms | 19.4 | 100% | 42.1% |
| 5 | 8 | NVDA | Nvidia | 156.7 | **81.7%** | 69.1% |
| 6 | 9 | V | Visa | 65.1 | 100% | 90.2% |
| 7 | 10 | WMT | Walmart | 140.6 | 100% | 98.3% |
| 8 | 16 | XOM | ExxonMobil | 118.2 | 100% | 91.5% |
| 9 | 18 | INTC | Intel | 146.7 | 100% | 97.6% |
| 10 | 22 | ORCL | Oracle | 46.2 | 100% | 76.4% |
| 11 | 23 | KO | Coca-Cola | 73.4 | 100% | 91.4% |
| 12 | 25 | CVX | Chevron | 106.8 | 100% | 95.6% |
| 13 | 27 | MRK | Merck | 75.3 | 100% | 90.4% |
| 14 | 28 | PEP | PepsiCo | 51.1 | 100% | 82.1% |
| 15 | 30 | COST | Costco | 63.5 | 100% | 86.2% |
| 16 | 31 | ABBV | AbbVie | 60.1 | 100% | 86.6% |
| 17 | 32 | ADBE | Adobe | 53.6 | 100% | 81.9% |
| 18 | 35 | TMO | Thermo Fisher | 31.5 | 100% | 66.5% |
| 19 | 36 | ABT | Abbott Laboratories | 43.2 | 100% | 79.0% |
| 20 | 37 | CRM | Salesforce | 73.8 | 100% | 88.6% |
| 21 | 38 | NKE | Nike | 63.0 | 100% | 89.0% |
| 22 | 40 | ACN | Accenture | 32.5 | 100% | 67.0% |
| 23 | 42 | DHR | Danaher | 27.9 | 100% | 62.4% |
| 24 | 43 | AMD | Advanced Micro Devices | 111.5 | 100% | 96.0% |
| 25 | 44 | NEE | NextEra Energy | 54.1 | 100% | 83.6% |
| 26 | 45 | CMCSA | Comcast | 54.5 | 100% | 86.5% |
| 27 | 46 | TXN | Texas Instruments | 43.5 | 100% | 73.4% |
| 28 | 48 | PYPL | PayPal | 78.0 | 100% | 89.9% |
| 29 | 49 | WFC | Wells Fargo | 80.2 | 100% | 91.2% |
| 30 | 50 | QCOM | Qualcomm | 68.0 | 100% | 88.7% |

Files: `data/universe/final_30_tickers_v2.csv` (list), `data/universe/screening_report.csv` (original 458-candidate gate), `data/universe/rescreen_report.csv` (replacement screen).

Composition: ~10 tech hardware/software, 3 communication services (GOOG, META, CMCSA), 2 payments (V, PYPL), 2 consumer discretionary, 4 consumer staples, 5 healthcare, 2 energy, 1 utility, 1 bank. No industrials, materials, or real estate. No sector cap by design (2.9).

#### 2.1.2 Universe Notes
- **Selection is partly driven by FNSPID availability.** The final 30 are large caps with continuous FNSPID coverage, not the 30 largest. Twelve large caps were screened out because FNSPID carries them in only ~30% of months: BRK.B, JNJ, UNH, MA, PG, HD, LLY, AVGO, BAC, PFE, VZ, MCD (plus LIN, 1.7%).
- **NVDA is a deliberate exception at 81.7% month coverage** (below the 85% gate). It has 11 missing months: 2020-07, 2020-08, 2020-10 and an unbroken block 2020-12 → 2021-07. This sits inside a thin patch of FNSPID itself (mid-2020 to 2021 is thin across many tickers), not an NVDA-specific fault. NVDA is kept because it is central to the AI-era market. Treat NVDA's gap months as price-only observations (has_news = 0). Planned robustness check: rerun key results with and without NVDA's gap months.
- **META's news is mostly NYT (63%).** FNSPID only carries Facebook/Meta in Feb–Jun 2020 (866 raw rows → 430 after dedupe; tagged `FB`, mapped to `META`). NYT supply fades after 2021 (≈16–17 articles/month in 2019–2021 vs ≈6.7 in 2022 and ≈5.4 in 2023). The article count jumps in Mar–May 2020 because of the source switch — an artifact; avoid raw counts for META.
- **Low day coverage despite full month coverage** for META (42.1% of days), AMZN (49.2%), AAPL (52.7%).
- GOOGL excluded as the second share class of Alphabet.
- Market-cap ranking uses a static-shares proxy (not true historical market cap); it only orders the screen. 26 tickers delisted/renamed after 2023 (e.g. WBA, DISH, ANSS) could not be ranked and rank last; they were never reached.
- Corporate-action check: Danaher's Veralto spin-off (ex-date 2023-10-02) shows no price cliff in the adjusted series (Yahoo appears to have adjusted for it).

#### 2.1.3 Known Limitations
1. Survivorship: 458 of 500 Jan-2019 constituents survive the continuous-listing filter. Shorts look less profitable and longs more profitable than in a full universe. Do not claim "no survivorship bias".
2. Market-cap screen uses the 2019-2023 average close x current shares (look-ahead in the screen; mild tilt toward growers). It only orders the screen.
3. Selection by news availability: BRK.B, JNJ, UNH, MA, PG, HD, LLY, AVGO, BAC, PFE, VZ, MCD (plus LIN) screened out for thin FNSPID coverage.
4. News is effectively one source: FNSPID 98.2% (139,075), NYT 1.8% (2,592). Sentiment = FNSPID headline sentiment in practice (META is the exception, 63% NYT).
5. Uneven coverage: 18.6% of ticker-days have no news; low day coverage for META (42.1%), AMZN (49.2%), AAPL (52.7%); NVDA has 11 gap months; FNSPID volume rises by year (2019: 21,045; 2023: 46,489).
6. Sector skew: ~10 tech names, 1 bank, 1 utility, no industrials/materials/real estate.
7. Efficient mega caps only: a null result does not say sentiment is useless in smaller stocks.
8. Short regime-specific sample: ~245 weekly cross-sections of 30 stocks (after dropping the first ~60 trading days of 2019); minimum detectable IC difference roughly 0.02-0.034 (to be confirmed by power simulation).
9. FNSPID timestamps are date-only, assigned t+1 (conservative: understates sentiment value).
10. Prices from yfinance (adjusted); macro series as-currently-published with approximate release lags; FinBERT trained on financial sentences, not headlines (validated on 200 hand-labeled headlines).
11. Sector labels use one static GICS snapshot (V and PYPL were reclassified in 2023).
12. Simplified trading assumptions: flat costs, flat borrow rate, no market impact, shorting always available.
13. News starts 2019-01-01 (FNSPID/NYT pulled for 2019–2023 only), so sentiment features have no history before 2019 (see 2.3.3).

**Claim scope:** among 30 well-covered, 2019-era large-cap survivors over 2019-2023, does FinBERT headline sentiment improve weekly ranking beyond price features?

### 2.2 Study Timeline
- **2019–2023** (revised down from 2016–2026). Labels, evaluation and backtests all live inside this window (the first ~60 trading days of 2019 are excluded from modeling, see 2.3.3).
- Deliberately includes the 2020 COVID crash/recovery and 2022 rate-hike selloff — results may be regime-dependent.

### 2.3 Data Sources
| Type | Source | Notes | Status |
|---|---|---|---|
| Price (OHLCV) | yfinance | Daily, `auto_adjust=True`; 30 tickers × 1,258 trading days (2019-01-02 → 2023-12-29) = 37,740 rows | **Done** — `data/prices/ohlcv_daily.csv` |
| News | FNSPID + NYT Archive API | 141,667 articles: 139,075 FNSPID (98.2%) + 2,592 NYT (1.8%) | **Done** — `data/news/news_final30.csv` |
| Macro | FRED API | 9 series, daily, release-lag aware | **Done** — `data/macro/fred_daily.csv` |

#### 2.3.1 News Pipeline (decisions)
- **Text fed to FinBERT:** FNSPID = `Article_title` only (title is 100% filled; the summary field is only 79% filled and 0% for some tickers, which would bias sentiment across tickers). NYT = headline + abstract (snippet fallback). The FNSPID summary is kept in a separate `summary_extra` column for later sensitivity checks. Text lengths are short (FNSPID median 58 chars, max 410; NYT median 205, max 656), so there is no FinBERT truncation.
- **FNSPID extraction:** both CSVs streamed, final-30 tickers, 2019–2023; `FB` mapped to `META`. 28.6M rows scanned.
- **FNSPID cleaning:** drop titles <15 chars; dedupe by (ticker, URL), then (ticker, day, normalized title). Removed ~10.7% of rows (some tickers had every article duplicated).
- **NYT matching (rewritten):** the original substring matcher produced false positives ("San Francisco"→Cisco, "intelligence"→Intel, "pineapple"→Apple, "visas"→Visa) and inflated the first coverage gate. Replaced with curated, case-sensitive, word-boundary patterns per ticker. An article is extracted if the company appears in the headline, abstract/snippet or the organizations tag; after cleaning, **only headline/abstract matches are kept** (tag-only rows are saved to `nyt_dropped_orgtag_only.csv`). News sections only (Business Day, Technology, U.S., World, Health, Climate, Science, Politics, New York, Washington); op-eds, reviews, letters, obits, videos, briefings dropped.
- **No-lookahead alignment:** each article gets a `feature_date` = the first trading day it may be used, assuming prediction at the market close.
  - FNSPID timestamps are date-only (99.6% midnight) → conservative next-trading-day rule (t+1). This works against the sentiment hypothesis (same-day reactions are lost), so a positive result is conservative.
  - NYT has real timestamps → converted to US Eastern; published before 16:00 ET on a trading day → same day; after 16:00 ET, weekend or holiday → next trading day. NYT assigned same-day: 63.2%.
  - 14 FNSPID rows at the very end of December 2023 fall past the last trading day and are dropped.
- **Coverage indicator:** `data/news/news_daily_coverage.csv` (trading day × ticker: n_articles, n_fnspid, n_nyt, has_news). 81.4% of ticker-days have news; 18.6% have none and keep score 0 with has_news = 0 (no synthetic imputation).
- **FNSPID volume is uneven by year** (2019: 21,045 · 2020: 23,131 · 2021: 18,422 · 2022: 29,988 · 2023: 46,489). Raw article-count features would encode the year; use has-news flags, mean sentiment, or per-ticker-normalized counts.

#### 2.3.2 Macro Series
| Column | FRED ID | Description | Native freq | Release handling |
|---|---|---|---|---|
| vixcls | VIXCLS | VIX | Daily | as observed (known at close) |
| dgs10 | DGS10 | 10-year Treasury yield | Daily | as observed |
| dgs2 | DGS2 | 2-year Treasury yield | Daily | as observed |
| baa10y | BAA10Y | Baa corporate yield spread over 10Y Treasury | Daily | as observed |
| dtwexbgs | DTWEXBGS | Broad USD trade-weighted index | Daily | as observed |
| dcoilwtico | DCOILWTICO | WTI crude oil (goes negative on 2020-04-20 — real; avoid log/percent transforms) | Daily | as observed |
| cpi_yoy | CPIAUCSL | CPI **year-over-year %**, computed from the index | Monthly | available ≈15th of next month (+14 days) |
| unrate | UNRATE | Unemployment rate | Monthly | available ≈8th of next month (+7 days) |
| fedfunds | FEDFUNDS | Effective fed funds rate (monthly average) | Monthly | available ≈2nd of next month (+1 day) |

- 1,826 calendar days × 9 indicators, 0 missing cells; fetched from 2017 so YoY and release shifts have history; trimmed to 2019-01-01..2023-12-31.
- BAA10Y chosen over ICE BofA high-yield spread (BAMLH0A0HYM2), whose FRED history is a rolling ~3-year window.
- Release lags are conservative approximations, not exact BLS dates. Values are as-currently-published (later revisions not reconstructed).

#### 2.3.3 Warm-up Data (planned)
- **Why:** rolling features (60-day beta, SMA50, 60-day return, 20-day macro changes) need history before the first label date. Warm-up data is used only to compute features; it never creates labels or evaluation rows.
- **Prices:** re-fetch from 2018-07 into a **separate file** (e.g. `data/prices/ohlcv_warmup.csv`) so the Phase 1 QA window check (2019-01-01..2023-12-31) on `ohlcv_daily.csv` stays valid.
- **Macro:** keep an untrimmed copy from 2017 for the 20-day changes and the curve slope.
- **News:** FNSPID/NYT are pulled for 2019–2023 only; there is no 2018 news. Sentiment features that use a trailing 60-day window (z-score, abnormal count) are not fully formed in early 2019. **Decision (2026-10-13):** the first ~60 trading days of 2019 are dropped from all tiers (F0/F1/F2), using a single `FIRST_MODEL_DATE` (the 61st trading day of 2019). This keeps paired F1 vs F0 comparisons on identical dates.

### 2.4 Feature Set
- **Technical (all stationary, scale-free):** returns over 1/5/10/20/60 trading days; close/SMA20 - 1; close/SMA50 - 1; MACD/close; RSI; 20-day realized volatility; log(volume / 20-day average volume); rolling 60-day beta to the equal-weight universe return. No raw price or volume levels.
- **Sentiment (6):** sent_mean, sent_ewm5 (half-life ~3 days), sent_mom (3-day ewm minus 20-day ewm), abn_count (log(1+n) z-scored vs the ticker's trailing 60 days), has_news, days_since_news (capped at 10). Article score = p_pos - p_neg. No synthetic imputation: no-news days keep score 0 and are flagged.
- **Macro (stationary):** 5-day change in VIX; 5-day change in 10Y; curve slope (10Y - 2Y) and its change; 20-day change in BAA10Y; 20-day return of WTI and of the USD index; CPI, unemployment and fed funds changes. Z-scored over time on training folds only.
- **Macro x stock interactions (8, pre-specified):** {dVIX, d10Y, dBAA10Y, WTI 20d return} x {rolling beta, 20d volatility}. No sector dummies.
- **Tiers:** F0 = technical (incl. beta, vol); F1 = F0 + sentiment; F2 = F1 + macro + 8 interactions.
- **Normalization:** technical and sentiment features rank-normalized per date across the 30 stocks (centered, about -0.5 to +0.5); binary flags unchanged; sentiment score features first z-scored per ticker over a trailing 60 days. Macro is not cross-sectionally ranked. Rolling windows use past data only; prices fetched from 2018-07 for warm-up (see 2.3.3). Scalers are fit inside training folds. Identical normalized inputs for every model.
- **Targets:** `target_ret5` = adj close t -> t+5 trading days (evaluation and P&L only). `target_rank` = per-date centered percentile rank, (rank - 0.5)/30 - 0.5 (primary training target, all models). `target_ret5_xs` = cross-sectionally demeaned, volatility-scaled return (XGBoost robustness only). Last 5 days of 2023 and dates with fewer than 30 valid labels are dropped.
- **Sentiment model:** ProsusAI/finbert (pinned revision), max_length 160, fp16, run once and cached; no fine-tuning; validated on 200 hand-labeled headlines (target macro F1 about 0.65; Cohen's kappa if two labelers); a second scorer only as a robustness run if validation fails.

### 2.5 Ablation Design

Two independent axes — feature tier (what information is available) and model family (how it's learned) — instead of one entangled matrix.

| Group | Model | F0: Price only | F1: +Sentiment | F2: +Sentiment+Macro |
|---|---|:---:|:---:|:---:|
| Baseline | Ridge | ✓ | ✓ | — |
| Tree ensemble | Random Forest | ✓ | ✓ | — |
| Tree ensemble | XGBoost | ✓ | ✓ | ✓ |
| Deep learning | LSTM | ✓ | ✓ | — |
| Transformer* | TFT | — | — | ✓ (optional stretch) |

9 core runs + 1 optional stretch run. F0→F1 comparisons across every family test whether the sentiment effect holds regardless of model type; F1→F2 on XGBoost isolates the macro contribution.

### 2.6 Evaluation

**Cross-validation**
- CPCV, date-based groups: N = 8, k = 2 (28 splits, 7 paths). Purge 5 trading days (label horizon); embargo 20 trading days (>= LSTM lookback); 60-day embargo sensitivity on XGBoost. Same splits and seeds for every model and tier; unit test that no training label window overlaps a test date.
- No separate holdout (power; 2023 holds about 33% of FNSPID volume). Safeguards: frozen design, any post-hoc run logged as a new trial, yearly results exploratory.
- Nested tuning: inner 4-fold purged blocks (LSTM: 2), metric = weekly rank IC, identical search budget per tier, small regularized grids, 3 seeds averaged for stochastic models.

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

### 2.7 Technology Stack
- **Modeling:** Python, PyTorch, XGBoost, scikit-learn, HuggingFace Transformers (FinBERT)
- **Experiment tracking:** MLflow only (SQLite backend, local artifacts). Parent run per model x tier x variant; child run per CPCV split. Tags: trial_id, trial_family, counts_toward_N. Log git hash and data checksums. Never delete runs. Commit a milestone CSV export (to a tracked folder such as `results/`, not under the git-ignored `data/`). W&B removed.
- **Serving:** read-only viewer of frozen precomputed results: PostgreSQL -> FastAPI -> React, local only. Four pages: Results, Backtest, Rankings, Explainability/Exposures. Built after Phase 5 freeze, capped at about 10% of project time, first item cut if behind schedule. Write the FastAPI endpoints and schema yourself; Lovable for React scaffolding only.
- **LSTM spec:** 20-day lookback, date-batched, hidden {32, 64}, 1-2 layers, dropout {0.2, 0.4}, AdamW, early stopping (patience 5, max 50 epochs), 3 seeds, about 1 GPU-day cap.
- **Run order:** Ridge -> XGBoost -> Random Forest -> LSTM -> robustness -> TFT (stretch, outside primary family).
- **Deployment:** none (Railway/Vercel dropped with Phase 7).

### 2.8 Model Matrix Addendum
TFT (E6) is kept as an optional stretch goal, run only if time permits after the 9 core experiments are complete and evaluated. It never gates the main conclusion.

### 2.9 Sector Composition
No explicit sector cap — the market-cap ranking + coverage gate decide the final mix naturally (composition recorded in 2.1.1). This is simpler to defend (one clear selection rule, not two competing ones) and avoids forcing in a thinly-covered stock just to fill a sector quota.

---

## 3. Phase Plan

| Phase | Focus | Status |
|---|---|---|
| 1 | Data collection & preprocessing | **Complete** (2026-10-07), QA 31/31 |
| 2 | Feature engineering (technical, FinBERT sentiment, macro) | **Next** |
| 3 | Model development (ablation matrix, MLflow tracking) | Planned |
| 4 | Evaluation (CPCV, IC, significance, SHAP) | Planned |
| 5 | Backtesting (long-short vs equal-weight 30, F0, random, reversal baselines) | Planned |
| 6 | Read-only results dashboard (FastAPI + React, local) | Planned, built last, first to cut |
| 7 | Cloud deployment | **Cut from scope** |

---

## 4. Open Decisions

- [ ] Static sector mapping for the 30 tickers (groups >= 3 stocks) and snapshot date
- [ ] Hand-label the 200-headline FinBERT validation sample (and decide second labeler)
- [ ] Finalize exact technical feature list and check correlations (drop > 0.95)
- [ ] Power simulation to confirm the minimum detectable IC difference
- [ ] Set the end-of-Phase-3 deadline date for the LSTM fallback rule
- [ ] Confirm Ken French factor data covers 2019-2023 and aligns to the trading calendar
- [ ] Re-fetch prices from 2018-07 for rolling-window warm-up (separate file, see 2.3.3) and keep untrimmed macro

Resolved: final 30 tickers; benchmark; target construction; CPCV configuration; NVDA robustness design; tracker; dashboard scope; first-60-day handling (dropped from all tiers).

---

## 5. Data Inventory

| Path | Contents |
|---|---|
| `data/news/fnspid/raw/` | `All_external.csv`, `nasdaq_exteral_data.csv` (FNSPID) |
| `data/news/fnspid/processed/` | `fnspid_monthly_counts.csv`, `fnspid_summary.csv`, `text_field_fill_rates.csv`, `fnspid_final30.csv` |
| `data/news/nytarchive/raw/` | Monthly NYT Archive JSON, 2019-01 → 2023-12 (258,030 docs) |
| `data/news/nytarchive/processed/` | `nytarchive_availability.csv`, `nytarchive_monthly_counts.csv`, `nytarchive_summary.csv`, `nytarchive_final30.csv`, `nyt_dropped_orgtag_only.csv` |
| `data/news/` | `news_clean_final30.csv`, `news_aligned_final30.csv`, **`news_final30.csv` / `.parquet` (final, 141,667 rows, unique `news_id`)**, `news_daily_coverage.csv`, `news_coverage_report.csv`, `cleaning_report.csv` |
| `data/universe/` | `candidates.csv` (458 ranked survivors), `screening_report.csv`, `rescreen_report.csv`, `final_30_tickers.csv` (superseded), **`final_30_tickers_v2.csv` (final)** |
| `data/prices/ohlcv_daily.csv` | 37,740 rows (30 tickers × 1,258 days) |
| `data/macro/fred_daily.csv` | 1,826 days × 9 indicators |

**Final news table schema:** `news_id, feature_date, ticker, source, text_type, text, published_utc, outlet, url, match_tier, assignment_rule`.

### Pipeline scripts (`fetch_and_validate/`)
| Script | Purpose |
|---|---|
| 1, 2, 3 | FNSPID coverage analysis; NYT Archive fetch; NYT availability check |
| 4, 5 | Original NYT matching and coverage gate (**superseded**: substring matching inflated counts) |
| 6 | FRED macro fetch (v4: CPI YoY + release lags) |
| 7 | OHLCV fetch (takes `--tickers`, `--start`, `--end`, `--out`) |
| 8 | FNSPID text-field fill-rate check |
| 9 | FNSPID extraction for final tickers |
| 10 | NYT extraction (curated patterns, tiered matching) |
| 11 | Clean, dedupe, relevance-filter |
| 12 | Align to trading days (its lag diagnostic counts weekend articles as same-day; use script 16 for authoritative figures) |
| 13 | Build final news table (adds `news_id`) |
| 14 | Coverage report + daily coverage panel |
| 15 | Re-screen universe and pick replacements |
| 16 | Phase 1 QA (31 checks incl. independent no-lookahead re-check) |

Re-run order for a new ticker list: 7 → 9 → 10 → 11 → 12 → 13 → 14 → 16.

---

## 6. Decision Log

The long-form log with reasoning lives in `DECISION_LOG.MD`. Summary:

| Date | Change |
|---|---|
| 2026-09-26 | Dropped NS1/NS2/NS3 source ablation; merged to single FNSPID+NYT source |
| 2026-09-26 | Stock universe reduced from 50 to 30 |
| 2026-09-26 | Timeline changed from 2016–2026 to 2019–2023 |
| 2026-09-26 | Restructured ablation matrix into feature-tier × model-family axes |
| 2026-09-26 | Phases 6–7 scoped down to a local-only dashboard; cloud deployment cut |
| 2026-09-26 | TFT (E6) confirmed as stretch goal, not required for core conclusion |
| 2026-09-26 | No sector cap — coverage gate + market cap ranking decide final mix |
| 2026-09-27 | Market-cap ranking switched to batched download × shares-outstanding proxy; 26 delisted/renamed tickers left unranked |
| 2026-09-28 | Macro set expanded from 5 to 9 FRED series; BAA10Y replaces BAML high-yield spread; DTWEXBGS spelling corrected |
| 2026-09-29 | Daily OHLCV confirmed as price granularity and fetched/validated |
| 2026-09-29 | FinBERT text: FNSPID title only; NYT headline + abstract |
| 2026-09-29 | Macro: CPI converted to YoY; CPI/UNRATE/FEDFUNDS shifted to approximate release dates |
| 2026-09-29 | NYT matching rewritten (word-boundary, case-sensitive, tiered) after discovering substring false positives |
| 2026-10-02 | Cleaning rules fixed (tag-only NYT matches dropped; news sections only; dedupe); trading-day alignment rule fixed (FNSPID t+1, NYT 16:00 ET) |
| 2026-10-02 | Coverage gate re-run on cleaned data: CSCO, JPM, NVDA failed; MSFT, NFLX, DIS <100% months |
| 2026-10-07 | Strict 100%-month rule applied (NVDA exception): CSCO, JPM, MSFT, NFLX, DIS replaced by CMCSA, TXN, PYPL, WFC, QCOM; pipeline re-run |
| 2026-10-07 | Phase 1 QA passed 31/31; Phase 1 closed |
| 2026-10-09 | Backtest benchmark changed from SPY to equal-weight 30-stock universe; F0 model, random ranking and 5-day reversal rule added as baselines. SPY dropped because the universe is not representative of the index |
| 2026-10-09 | Pre-registered hypotheses H1 (F1>F0) and H2 (F2>F1, XGBoost); H1 requires BH-adjusted p < 0.05 and CI excluding zero in >= 3 of 4 families; null results reported as findings |
| 2026-10-09 | Significance: weekly non-overlapping rank IC; paired difference via block bootstrap (primary) and HAC/DM with HLN (cross-check); BH over 5 primary tests; power analysis before modeling |
| 2026-10-09 | Targets: target_ret5 (evaluation/P&L), target_rank (primary training target), target_ret5_xs (XGBoost robustness); backtest tests 1-day execution delay |
| 2026-10-09 | Features: stationary technical ratios; per-date cross-sectional rank; macro z-scored on training folds only; prices from 2018-07 for warm-up; identical inputs for all models |
| 2026-10-09 | Sentiment: 6 daily features; per-ticker 60d z-score then cross-sectional rank; no synthetic imputation; FNSPID t+1 kept; robustness without META and without NVDA gap months |
| 2026-10-09 | FinBERT: ProsusAI/finbert pinned, max_length 160, fp16, run once, no fine-tuning; 200-headline hand-label validation; second scorer only as robustness |
| 2026-10-11 | Macro (F2): stationary signals plus 8 pre-specified macro x {beta, vol} interactions; beta and vol added to F0; no sector dummies; H2 flagged as exploratory/likely null |
| 2026-10-13 | CPCV: N=8, k=2 (28 splits, 7 paths); purge 5, embargo 20 trading days; 60-day embargo sensitivity; per-date IC averaged across paths |
| 2026-10-13 | No separate holdout (power; ~250 weekly cross-sections; 2023 holds ~33% of FNSPID volume). Safeguards: frozen design, post-hoc changes logged as new trials. Nested tuning: inner 4-fold purged blocks (LSTM 2), metric = weekly rank IC |
| 2026-10-13 | Metrics: primary = weekly rank IC; secondary = ICIR, hit rate, quintile spread, OOS R2, net Sharpe, drawdown, turnover, break-even cost, beta/alpha; directional accuracy dropped |
| 2026-10-13 | Portfolio: weekly rebalance, 5 staggered offsets; dollar-neutral top/bottom quintile (6/6); secondary long-only top quintile and rank-weighted variants; entry at close t; per-path backtests |
| 2026-10-13 | Costs: base 5 bps/side + 0.5% borrow; grid 0/5/10/20 bps and 0.25/0.5/1/3%; Feb-Apr 2020 3x stress; break-even cost; optional buffer variant |
| 2026-10-13 | Exposure diagnostics added (market, Fama-French + MOM + ST reversal, sector, concentration); beta-neutral variant if |beta| > 0.15 or significant |
| 2026-10-13 | Multiple testing: BH q=0.05 on 5 primary IC tests, BY check, exploratory family separate; deflated Sharpe with tracker-counted N, bar DSR >= 0.95 |
| 2026-10-13 | Experiment tracking: MLflow only; W&B dropped |
| 2026-10-13 | Dashboard: read-only frozen results viewer, 4 pages, local, built after Phase 5, ~10% time cap |
| 2026-10-13 | LSTM spec frozen with ~1 GPU-day cap and time-based fallback (H1 becomes >= 2 of 3 if not done by end of Phase 3); TFT stretch outside primary family; XGBoost-only robustness list |
| 2026-10-13 | Limitations documented (2.1.3); claims scoped to this universe and period |
| 2026-10-13 | Warm-up data: 2018-07 prices (separate file) and untrimmed 2017+ macro used only for rolling-feature warm-up; news stays 2019–2023 |
| 2026-10-13 | First ~60 trading days of 2019 dropped from ALL tiers (F0/F1/F2) via a single FIRST_MODEL_DATE (61st trading day of 2019), so paired F1-vs-F0 tests use identical dates |
