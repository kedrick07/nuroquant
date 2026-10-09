# NuroQuant — Project Scope & Reference Document

**Author:** Kedrick Selvanesan · Dept. of Multimedia, FSKTM, UPM
**Supervisor:** Assoc. Prof. Dr. Azreen bin Azman
**Type:** Final Year Project

This document is the single source of truth for project scope. When a decision changes, update it here first — everything else (code, thesis chapters, Gantt chart) should follow this file, not the other way around.

**Status (2026-10-07): Phase 1 (data collection & preprocessing) is complete. Phase 1 QA: 31/31 checks passed. Next: Phase 2 (feature engineering).**

---

## 1. Core Research Question

> Does news sentiment (FinBERT-derived) measurably improve short-term equity ranking performance beyond price/volume features alone — and does adding macroeconomic context improve it further?

Secondary framing: NuroQuant predicts continuous 5-day forward returns and converts them into a **cross-sectional ranking** of a large-cap stock universe, rather than issuing buy/sell/hold classifications.

---

## 2. Finalized Scope

### 2.1 Stock Universe
- **Size:** 30 stocks (revised down from 50)
- **Selection method:**
  1. S&P 500 constituents as of **Jan 2019** (not today's list — avoids survivorship bias)
  2. Keep tickers continuously listed through **2023** (no delistings, mid-window IPOs, or major spin-offs/mergers) → 458 survivors
  3. Rank survivors by average market cap over the window (proxy: mean daily close 2019–2023 × current shares outstanding; used only to order the screen)
  4. **Coverage gate**, applied to the *cleaned* news data (FNSPID + NYT): average ≥5 articles/month AND ≥85% of the 60 months with ≥1 article
  5. **Reliability filter** (strict): keep only tickers with 100% month coverage, replacing failures with the next-ranked ticker that has 100% FNSPID month coverage. **One documented exception: NVDA** (see 2.1.2)
  6. Same-company duplicate share classes excluded (GOOGL; GOOG kept)
- **Status:** Locked (2026-10-07). See 2.1.1.

#### 2.1.1 Final 30 Tickers

Coverage numbers are from the cleaned, trading-day-aligned news table (`news_coverage_report.csv`). "Days with news" is the share of the 1,258 trading days with ≥1 article; the rest are neutral-imputed and flagged.

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

#### 2.1.2 Known Limitations of the Universe
- **Selection is partly driven by FNSPID availability.** The final 30 are large caps with continuous FNSPID coverage, not the 30 largest. Twelve large caps were screened out because FNSPID carries them in only ~30% of months: BRK.B, JNJ, UNH, MA, PG, HD, LLY, AVGO, BAC, PFE, VZ, MCD (plus LIN, 1.7%). State this in the thesis.
- **NVDA is a deliberate exception at 81.7% month coverage** (below the 85% gate). It has 11 missing months: 2020-07, 2020-08, 2020-10 and an unbroken block 2020-12 → 2021-07. This sits inside a thin patch of FNSPID itself (mid-2020 to 2021 is thin across many tickers), not an NVDA-specific fault. NVDA is kept because it is central to the AI-era market. Treat NVDA's gap months as price-only observations (has-news flag = 0). Planned robustness check: rerun key results with and without NVDA's gap months.
- **META's news is mostly NYT (63%).** FNSPID only carries Facebook/Meta in Feb–Jun 2020 (866 raw rows → 430 after dedupe; tagged `FB`, mapped to `META`). NYT supply fades after 2021 (≈16–17 articles/month in 2019–2021 vs ≈6.7 in 2022 and ≈5.4 in 2023). The article count jumps in Mar–May 2020 because of the source switch — an artifact; avoid raw counts for META.
- **Low day coverage despite full month coverage** for META (42.1% of days), AMZN (49.2%), AAPL (52.7%).
- GOOGL excluded as the second share class of Alphabet.
- Market-cap ranking uses a static-shares proxy (not true historical market cap); it only orders the screen. 26 tickers delisted/renamed after 2023 (e.g. WBA, DISH, ANSS) could not be ranked and rank last; they were never reached.
- Corporate-action check: Danaher's Veralto spin-off (ex-date 2023-10-02) shows no price cliff in the adjusted series (Yahoo appears to have adjusted for it).

### 2.2 Study Timeline
- **2019–2023** (revised down from 2016–2026)
- Deliberately includes the 2020 COVID crash/recovery and 2022 rate-hike selloff — acknowledge in limitations that results may be regime-dependent.

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
- **NYT matching (rewritten):** the original substring matcher produced false positives ("San Francisco"→Cisco, "intelligence"→Intel, "pineapple"→Apple, "visas"→Visa) and inflated the first coverage gate. Replaced with curated, case-sensitive, word-boundary patterns per ticker. An article is extracted if the company appears in the headline, abstract/snippet or the organizations tag; after cleaning, **only headline/abstract matches are kept** (tag-only rows are saved to `nyt_dropped_orgtag_only.csv`, because FinBERT would score text that never mentions the company). News sections only (Business Day, Technology, U.S., World, Health, Climate, Science, Politics, New York, Washington); op-eds, reviews, letters, obits, videos, briefings dropped.
- **No-lookahead alignment:** each article gets a `feature_date` = the first trading day it may be used, assuming prediction at the market close.
  - FNSPID timestamps are date-only (99.6% midnight) → conservative next-trading-day rule (t+1). This works against the sentiment hypothesis (same-day reactions are lost), so a positive result is conservative.
  - NYT has real timestamps → converted to US Eastern; published before 16:00 ET on a trading day → same day; after 16:00 ET, weekend or holiday → next trading day. NYT assigned same-day: 63.2%.
  - 14 FNSPID rows at the very end of December 2023 fall past the last trading day and are dropped.
- **Coverage indicator:** `data/news/news_daily_coverage.csv` (trading day × ticker: n_articles, n_fnspid, n_nyt, has_news). 81.4% of ticker-days have news; **18.6% will be neutral-imputed and flagged**.
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

### 2.4 Feature Set
- **Technical:** moving averages, RSI, MACD, volatility (~12 columns)
- **Sentiment:** FinBERT pos/neg/neu probabilities, has-news flag, article count (normalized), sentiment momentum (~5 columns)
- **Macro:** the 9 series in 2.3.2
- **Targets:** `target_return_5d`, `target_rank`

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
- **Cross-validation:** Combinatorial Purged Cross-Validation (CPCV) with purging/embargo
- **Metrics:** Information Coefficient (IC), directional accuracy, Sharpe ratio, max drawdown
- **Interpretability:** SHAP feature attribution across the three feature groups
- **Significance:** paired t-tests with Benjamini–Hochberg correction across configurations
- **Backtest:** long-short decile portfolio, realistic trading costs, benchmarked against SPY buy-and-hold

### 2.7 Technology Stack
- **Modeling:** Python, PyTorch, XGBoost, scikit-learn, HuggingFace Transformers (FinBERT)
- **Experiment tracking:** MLflow, Weights & Biases
- **Serving:** FastAPI backend, React frontend (via Lovable AI), PostgreSQL — run locally
- **Deployment:** none (Railway/Vercel dropped with Phase 7)

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
| 3 | Model development (ablation matrix, MLflow/W&B tracking) | Planned |
| 4 | Evaluation (CPCV, IC, Sharpe, SHAP) | Planned |
| 5 | Backtesting (long-short portfolio vs. SPY) | Planned |
| 6 | Dashboard (FastAPI + React) | Minimal, local-only — no deployment |
| 7 | Cloud deployment | **Cut from scope** |

---

## 4. Open Decisions (Phase 2)

- [ ] Sentiment feature design: how to aggregate multiple articles per (ticker, feature_date); how to represent the 18.6% neutral-imputed days; count normalization (FNSPID volume skew by year; META source switch)
- [ ] FinBERT checkpoint and batching/hardware plan (≈141,667 short texts)
- [ ] Target construction details: 5-day forward return from the close of `feature_date`; cross-sectional rank definition
- [ ] CPCV configuration (number of groups, purge/embargo length vs. the 5-day horizon)
- [ ] NVDA robustness check design (with vs. without gap months)

Resolved in Phase 1: final 30 tickers; CPI representation (YoY); macro release-lag shifts; FNSPID/NYT text fields.

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
| 7 | OHLCV fetch (takes `--tickers`) |
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
