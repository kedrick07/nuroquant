# CONTEXT.md — NuroQuant (Final Year Project)

> Purpose: ground-truth reference for AI assistants. Read this BEFORE answering anything about NuroQuant.
> Rules for the assistant:
> 1. If a fact is not in this file or in code the user pastes, say "not in context" — do not guess.
> 2. Never invent file names, column names, function signatures, package APIs or versions. Ask for the code or docs.
> 3. If this file conflicts with newer user info, trust the user and flag the conflict.
> 4. Do not change the frozen design (Section 6). Any change = new logged trial.
> 5. Style: user (Kedrick) wants short, simple, practical answers; he writes code himself, so guide and explain rather than dump full solutions unless asked.

Last synced: 2026-10-14 (Phase 1 closed, Phase 2 next). Sources: README.md (+ merged update patch), PHASE_1_PROGRESS_REPORT.md, PROGRESS_REPORT_TEMPLATE.md.
Update this file at the end of every phase.

---

## 1. Project Identity
- Name: NuroQuant. Type: Final Year Project (UPM, FSKTM, Dept. of Multimedia).
- Author: Kedrick Selvanesan. Supervisor: Assoc. Prof. Dr. Azreen bin Azman.
- Research question: Does FinBERT news sentiment improve short-term equity ranking beyond price/volume features, and does macro context improve it further?
- Output: continuous 5-day forward return predictions turned into a cross-sectional ranking of 30 stocks (NOT buy/sell/hold classification).
- Claim scope: among 30 well-covered, 2019-era large-cap survivors over 2019–2023, does FinBERT headline sentiment improve weekly ranking beyond price features?

## 2. Current Status
- Phase 1 (data): COMPLETE. QA `16_phase1_qa.py` = 31/31 pass. One documented exception: NVDA coverage (81.7%).
- Git: `main` = `dev` at `ae19c97` ("fully completed phase 1"). Phase 2 work goes on `dev`; merge to `main` when tested. No tag `phase1-complete` yet.
- Next: Phase 2 (feature engineering).
- Still unchecked on Phase 1 sign-off: data backup + checksums; optional tag; hand-off notes review.

## 3. Phase Plan
| Phase | Focus | Status |
|---|---|---|
| 1 | Data collection & preprocessing | Done |
| 2 | Feature engineering (technical, FinBERT sentiment, macro) | Next |
| 3 | Models (ablation matrix, MLflow) | Planned |
| 4 | Evaluation (CPCV, IC, SHAP, significance) | Planned |
| 5 | Backtesting (long-short vs baselines) | Planned |
| 6 | Read-only results dashboard (FastAPI + React, local) | Planned, built last, first to cut |
| 7 | Cloud deployment | CUT |

## 4. Data (Phase 1 outputs)
Headline numbers:
- Prices: 30 tickers x 1,258 trading days (2019-01-02 to 2023-12-29) = 37,740 rows, adjusted, from yfinance.
- News: 141,667 articles (FNSPID 139,075 = 98.2%; NYT 2,592 = 1.8%). 81.4% of ticker-days have news; 18.6% have none.
- Macro: 9 daily series, 1,826 calendar days, 0 NaN (FRED).
- Warm-up: 3,780 price rows (126 days, 2018-07-02 to 2018-12-31); 2,010 macro days from 2018-07-01. No 2018 news exists.

Files (all under `data/`, which is GIT-IGNORED — back up separately):
| Path | Contents | Use |
|---|---|---|
| `data/universe/final_30_tickers_v2.csv` | final 30 tickers (locked 2026-10-07) | ticker list |
| `data/prices/ohlcv_daily.csv` | 37,740 rows, 2019-01-02..2023-12-29 | features + labels + eval |
| `data/prices/ohlcv_warmup.csv` | 3,780 rows, 2018-07-02..2018-12-31 | feature warm-up ONLY |
| `data/macro/fred_daily.csv` | 1,826 days x 9 | QA only |
| `data/macro/fred_daily_full.csv` | 2,010 days x 9 (from 2018-07-01) | USE for macro features |
| `data/news/news_final30.csv` / `.parquet` | 141,667 rows, unique `news_id` | FinBERT input |
| `data/news/news_daily_coverage.csv` | n_articles, n_fnspid, n_nyt, has_news per ticker-day (37,740 rows) | coverage flags |

News schema (exact columns): `news_id, feature_date, ticker, source, text_type, text, published_utc, outlet, url, match_tier, assignment_rule`

NOT in context (ask user, do not guess): the actual 30 ticker symbols, exact price/macro CSV column names, the names of the 9 macro series beyond those listed here.
Known tickers mentioned: NVDA, META, AMZN, AAPL, CRM, DHR, V, PYPL, CMCSA, TXN, WFC, QCOM (added 2026-10-07 replacing CSCO, JPM, MSFT, NFLX, DIS). Screened out for thin FNSPID coverage: BRK.B, JNJ, UNH, MA, PG, HD, LLY, AVGO, BAC, PFE, VZ, MCD, LIN.

Macro series known: fed funds, CPI (as YoY), unemployment (UNRATE), 10Y yield, VIX, 2Y yield (curve slope 10Y-2Y), BAA10Y (credit spread, replaced BAML HY), WTI, USD index. (These are inferred from the feature spec; confirm exact FRED IDs with the user.)

Pipeline scripts: `fetch_and_validate/` scripts 1–16 (tracked in git). Re-run order for a new ticker list: 7 -> 9 -> 10 -> 11 -> 12 -> 13 -> 14 -> 16.

## 5. Commands (only ones documented)
```
python fetch_and_validate/7_fetch_price_ohlcv.py --start 2018-07-01 --end 2019-01-01 --out data/prices/ohlcv_warmup.csv
python fetch_and_validate/6_fetch_macro_fred.py      # writes fred_daily.csv AND fred_daily_full.csv
python fetch_and_validate/16_phase1_qa.py            # expect 31/31
```
Not recorded yet: Python/package versions, data checksums. Add in Phase 2 setup.

## 6. FROZEN Design (do not change without logging a new trial)

### 6.1 Hypotheses
- H1: F1 has higher weekly rank IC than F0. Supported if paired weekly IC diff (F1-F0) is significant (BH-adjusted two-sided p<0.05, bootstrap 95% CI excludes 0) in >=3 of 4 model families.
- H2: F2 > F1 (XGBoost only). Weaker hypothesis; null result plausible and reported as a finding.
- Fallback (time-based): if LSTM F0/F1 not done by end of Phase 3, drop LSTM from primary family; H1 then needs 2 of 3 families, BH over 4 tests. Deadline date NOT yet set.
- Primary metric: mean weekly rank IC on non-overlapping 5-day cross-sections, out of sample.

### 6.2 Feature tiers
- F0 = technical (incl. beta, vol). F1 = F0 + sentiment. F2 = F1 + macro + 8 interactions.
- Technical (stationary, scale-free; no raw price/volume levels): returns over 1/5/10/20/60 days; close/SMA20 - 1; close/SMA50 - 1; MACD/close; RSI; 20d realized vol; log(volume / 20d avg volume); rolling 60d beta to equal-weight universe return.
- Sentiment (6): `sent_mean`, `sent_ewm5` (half-life ~3d), `sent_mom` (3d ewm - 20d ewm), `abn_count` (log(1+n) z-scored vs ticker's trailing 60d), `has_news`, `days_since_news` (cap 10). Article score = p_pos - p_neg. No synthetic imputation: no-news days score 0 with has_news=0.
- Macro (stationary): 5d change VIX; 5d change 10Y; curve slope (10Y-2Y) and its change; 20d change BAA10Y; 20d return WTI and USD index; CPI, unemployment, fed funds changes. Z-scored over time on training folds only.
- 8 interactions: {dVIX, d10Y, dBAA10Y, WTI 20d return} x {rolling beta, 20d vol}. No sector dummies.
- Normalization: technical + sentiment rank-normalized per date across 30 stocks (centered, ~-0.5..+0.5); binary flags unchanged; sentiment scores first z-scored per ticker over trailing 60d. Macro NOT cross-sectionally ranked. Rolling windows past-only. Scalers fit inside training folds. Identical normalized inputs for every model.

### 6.3 Targets
- `target_ret5`: adj close t -> t+5 trading days (evaluation and P&L only).
- `target_rank`: per-date centered percentile rank, (rank - 0.5)/30 - 0.5. PRIMARY training target for all models.
- `target_ret5_xs`: cross-sectionally demeaned, vol-scaled return (XGBoost robustness only).
- Drop last 5 days of 2023 and dates with <30 valid labels.

### 6.4 FinBERT
`ProsusAI/finbert`, pinned revision, max_length 160, fp16, run once and cache, no fine-tuning. Text: FNSPID title only; NYT headline + abstract. Validate on 200 hand-labeled headlines (target macro F1 >= ~0.65; Cohen's kappa if two labelers). Second scorer only as robustness run if validation fails. `summary_extra` kept for F1-only robustness run.

### 6.5 Models and ablation (9 core runs + optional TFT)
| Model | F0 | F1 | F2 |
|---|:-:|:-:|:-:|
| Ridge | yes | yes | - |
| Random Forest | yes | yes | - |
| XGBoost | yes | yes | yes |
| LSTM | yes | yes | - |
| TFT (stretch, outside primary family) | - | - | optional |
Run order: Ridge -> XGBoost -> Random Forest -> LSTM -> robustness -> TFT.
LSTM spec: 20-day lookback, date-batched, hidden {32,64}, 1-2 layers, dropout {0.2,0.4}, AdamW, early stop (patience 5, max 50 epochs), 3 seeds, ~1 GPU-day cap.
Tuning: nested, inner 4-fold purged blocks (LSTM: 2), metric = weekly rank IC, identical search budget per tier, 3 seeds averaged for stochastic models.

### 6.6 Evaluation
- CPCV, date-based groups: N=8, k=2 (28 splits, 7 paths). Purge 5 trading days; embargo 20 trading days; 60-day embargo sensitivity on XGBoost. Same splits/seeds for all models and tiers. Unit test: no training label window overlaps a test date.
- No separate holdout (2023 holds ~33% of FNSPID volume). Post-hoc runs logged as new trials; yearly results exploratory.
- Significance: weekly non-overlapping rank IC; paired diff via block bootstrap (primary, 10,000 resamples) + HAC Diebold-Mariano with HLN (cross-check). Per-date IC averaged across 7 paths first. Primary family = 5 tests (H1 x 4 families + H2), BH q=0.05, BY as conservative check. Other tests = separate exploratory BH family. Power simulation BEFORE modeling (expected min detectable IC diff ~0.02–0.034, unconfirmed).
- Deflated Sharpe on weekly net Sharpe; N from tracker tags (core + robustness + variants; cost grids and nested tuning excluded); bar DSR >= 0.95.
- Secondary metrics: ICIR, IC hit rate, quintile returns/spread, OOS R2 on target_rank, net Sharpe (bootstrap CI), max drawdown, Calmar, turnover, break-even cost, beta/alpha vs EW. Directional accuracy DROPPED.
- Interpretability: group-level SHAP shares + group permutation importance.
- Robustness (XGBoost F0/F1 only, exploratory): mask NVDA gap months; exclude META; `target_ret5_xs`; 60-day embargo; add FNSPID `summary_extra` (F1 only).

### 6.7 Backtest
- Weekly (5-day) rebalance, 5 staggered start offsets averaged. Main: dollar-neutral long-short, top/bottom quintile (6/6 equal weight). Secondary: long-only top quintile vs EW-30; rank-weighted dollar-neutral. Entry at close t; 1-day-delay sensitivity. Run per CPCV path (7); report median and range.
- Benchmarks (SPY DROPPED): equal-weight 30-stock universe (primary), F0 model, random ranking (many seeds), 5-day reversal rule.
- Costs: base 5 bps/side + 0.5% annual borrow on shorts; grid 0/5/10/20 bps and 0.25/0.5/1/3% borrow; Feb–Apr 2020 3x stress; report break-even cost; optional buffer variant (enter top 20%, exit below 40%).
- Exposure diagnostics (exploratory): Newey-West regression on EW-30 return; Fama-French + momentum + ST reversal; weekly net sector exposure and sector-neutral IC (static GICS, groups >= 3); top-3 P&L concentration; beta-neutral variant if |beta|>0.15 or significant.

### 6.8 Tech stack
- Modeling: Python, PyTorch, XGBoost, scikit-learn, HuggingFace Transformers (FinBERT).
- Tracking: MLflow ONLY (SQLite backend, local artifacts). Parent run per model x tier x variant; child run per CPCV split. Tags: `trial_id`, `trial_family`, `counts_toward_N`. Log git hash + data checksums. Never delete runs. Commit milestone CSV export to a tracked folder (e.g. `results/`). W&B REMOVED.
- Serving: read-only viewer of frozen precomputed results: PostgreSQL -> FastAPI -> React, local only, 4 pages (Results, Backtest, Rankings, Explainability/Exposures). Capped ~10% of project time. User writes FastAPI endpoints + schema himself; Lovable for React scaffolding only. Railway/Vercel dropped.

## 7. Look-ahead and Leakage Rules (critical)
- FNSPID timestamps are date-only (99.6% midnight) -> assigned t+1 (`feature_date`).
- NYT: converted to US Eastern; after 16:00 ET, weekend or holiday -> next trading day.
- Macro: release lags applied (UNRATE ~+7d, CPI +14d, FEDFUNDS +1d); CPI converted to YoY. No extra lagging in Phase 2. Lags are approximations.
- Features at date t use only data up to t. Scalers fit inside training folds only.
- Warm-up files: features only, never labels or evaluation.
- All tiers filtered to the same dates from `FIRST_MODEL_DATE`.

## 8. Phase 2 Hand-off (what to build next)
Conventions:
- Create `features/config.py` with `FIRST_MODEL_DATE` (61st trading day of 2019; ~first 60 days dropped), paths, ticker list. NEVER hard-code a calendar date.
- Prices: concatenate `ohlcv_warmup.csv` + `ohlcv_daily.csv` on date+ticker. Macro: `fred_daily_full.csv`, calendar-daily, reindex to trading days. News: `news_final30.parquet` + `news_daily_coverage.csv`.
- ~245 weekly cross-sections remain after the cut.

Suggested order: (1) `features/` + `config.py`; (2) technical features (F0) + rolling beta; (3) FinBERT scoring/caching (slowest) while hand-labeling 200 headlines; (4) sentiment features; (5) macro features + 8 interactions; (6) targets, then unit tests.

Acceptance checks: no look-ahead; no NaNs on/after `FIRST_MODEL_DATE`; F0/F1/F2 share identical (date, ticker) rows; warm-up ends before 2019-01-02 with same 30 tickers; FinBERT macro F1 on 200 headlines >= ~0.65 (else queue 2nd scorer).

Open decisions: hand-label 200 headlines (+ second labeler?); finalize technical list and drop features correlated > 0.95; power simulation; static sector mapping + snapshot date; LSTM fallback deadline; confirm Ken French factors cover 2019–2023 and align to trading calendar.

## 9. DO-NOT-DO List
- No raw article counts as features (FNSPID volume grows by year: 2019 21,045; 2020 23,131; 2021 18,422; 2022 29,988; 2023 46,489). Use `has_news`, mean sentiment, per-ticker-normalized counts.
- No log/percent transforms on WTI (goes negative 2020-04-20; real).
- No warm-up data for labels/evaluation.
- No substring news matching (false positives: "San Francisco" -> Cisco, "intelligence" -> Intel, "pineapple" -> Apple, "visas" -> Visa). Use case-sensitive word-boundary patterns.
- No overwriting `README.md` from a stale copy. `git pull` before editing; check README first line after any merge.
- Don't say "no survivorship bias" (458 of 500 Jan-2019 constituents survive).
- Quote script 16 QA numbers, not script 12 (it over-reports same-day).
- Re-run the warm-up boundary price check if prices are ever re-fetched.
- Don't change alignment rules or frozen design without logging a new trial.

## 10. Data Quirks to Remember
- NVDA: 81.7% month coverage; 11 missing months (2020-07, 2020-08, 2020-10, 2020-12 to 2021-07) -> treat price-only (`has_news=0`).
- META: FNSPID tags it `FB` (Feb–Jun 2020) -> mapped to META; 63% NYT; day coverage 42.1%. AMZN 49.2%, AAPL 52.7% day coverage.
- Big moves kept as real: CRM +26% (2020-08-26), META -26% (2022-02-03). DHR Veralto spin-off (2023-10-02): no cliff in adjusted series.
- Warm-up join: 2018-12-31 -> 2019-01-02 moves -3.91% to +3.50%, none > 10%.
- Dedupe rules: FNSPID by (ticker, URL), then (ticker, day, normalized title); titles < 15 chars dropped (~10.7% removed). 14 FNSPID rows after last trading day dropped. NYT org-tag-only rows saved to `nyt_dropped_orgtag_only.csv`.
- Reliability filter: strict 100% month coverage (NVDA the only exception). Original 85% gate superseded.

## 11. Known Limitations (state in thesis)
Survivorship; news ~98% FNSPID (say "headline sentiment from FNSPID"); universe chosen partly by news availability; short regime-specific sample (COVID 2020, 2022 rate hikes); sector skew (~10 tech, 1 bank, 1 utility); efficient mega-caps only; FNSPID t+1 is conservative; market-cap screen uses current shares (look-ahead, selection only); macro as-currently-published; FinBERT trained on sentences not headlines; one static GICS snapshot (V and PYPL reclassified 2023); simplified trading assumptions (flat costs, always-shortable); data files git-ignored (backup risk).

## 12. Key Decision History (condensed)
- 2026-09-26: single FNSPID+NYT source; 50->30 stocks; 2016–2026 -> 2019–2023; ablation = feature tier x model family; cloud deployment cut; TFT stretch; no sector cap.
- 2026-09-27: market-cap ranking via batched download x shares proxy.
- 2026-09-28: macro 5 -> 9 series; BAA10Y replaces BAML HY spread.
- 2026-09-29: FinBERT text choice; macro release lags.
- 2026-10-07: strict 100% coverage; 5 ticker replacements.
- 2026-10-09: benchmark SPY -> EW-30; hypotheses pre-registered; significance design; targets.
- 2026-10-11: macro/sentiment/FinBERT specs.
- 2026-10-13: CPCV N=8,k=2; no holdout; costs; MLflow only; dashboard scope; LSTM spec; limitations; README rebuilt.
- 2026-10-14: warm-up data fetched; Phase 1 closed; dev merged to main.
Full log: `DECISION_LOG.MD` (not in this context; ask user to paste if needed).

## 13. Repo and Process
- Tracked: `fetch_and_validate/`, `README.md`, `DECISION_LOG.MD`, `.gitignore` (308 bytes). Ignored: `data/`.
- Branches: `main`, `dev`. Key commits: `fbfc6d2` (data scripts pushed), `81743819`, `2d00696`, `920979b` (README rebuild/restores), `68341ea`, `e1e9cc4`, `ae19c97` (Phase 1 close).
- README incident (E14): reverted twice via stale copy and patch overwrite. `README_update_patch.md` was merged and deleted.
- Phase reports: copy `PROGRESS_REPORT_TEMPLATE.md` to `reports/PHASE_<N>_PROGRESS_REPORT.md`; fill every section (write "None" rather than deleting). 14 sections 0–13: header, summary, objectives, deliverables, scope changes, errors, quality, limitations, open decisions, hand-off, reproducibility, timeline, repo notes, sign-off.

## 14. Gaps — Paste These Into Chat When Relevant
This file has NO code. For accurate syntax/API help, paste: the relevant script, `requirements.txt` / package versions, the CSV headers, `features/config.py`, and any traceback. Otherwise the assistant must say it is unsure instead of guessing.
