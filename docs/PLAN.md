# PLAN

Last updated: 2026-10-19

## Goal
Test, under a pre-registered design, whether FinBERT headline sentiment (H1) and macro context (H2) improve weekly cross-sectional ranking of 30 US large caps over 2019-2023.

## Claim scope
Among 30 well-covered, 2019-era large-cap survivors over 2019-2023, does FinBERT headline sentiment improve weekly ranking beyond price features? No wider claim.

## Non-goals
- No buy, sell or hold classification.
- No live trading, live data, inference service or auth.
- No cloud deployment (Phase 7 cut).
- No claims for small caps, other periods or other news sources.
- No SPY benchmark (dropped).
- TFT is a stretch goal and never gates the conclusion.

## Hypotheses
- H1: F1 (price + sentiment) beats F0 (price) on weekly rank IC in at least 3 of 4 model families (BH q=0.05, bootstrap CI excludes 0).
- H2: F2 (adds macro) beats F1, XGBoost only. Likely null; a null is reported as a finding.
- Fallback: if LSTM F0/F1 are not done by the end of Phase 3, H1 needs 2 of 3 families. Deadline date: NOT SET.

## Scope in numbers
- 30 tickers, 1,258 trading days, 141,667 articles (98.2% FNSPID), 9 FRED series.
- Model rows: 35,790 (1,193 dates, 2019-03-29 to 2023-12-21), about 239 weekly cross-sections.
- Simulated power: 80% power needs a true IC gain of about 0.015-0.032, so a null H1 is plausible.
- 9 core runs (Ridge, Random Forest, XGBoost, LSTM across tiers; see ../README.md section 2.5) plus optional TFT.

## Phases
| Phase | Focus | Status |
|---|---|---|
| 1 | Data collection and preprocessing | Done (QA 31/31) |
| 2 | Feature engineering | Built, QA 23/24, labels pending |
| 3 | Models, MLflow tracking | Next |
| 4 | Evaluation: CPCV, IC, significance, SHAP | Planned |
| 5 | Backtest vs equal-weight-30, F0, random, reversal | Planned |
| 6 | Read-only dashboard (FastAPI + React, local) | Planned, built last, first to cut |
| 7 | Cloud deployment | Cut |

## Frozen design
Hypotheses, features, targets, CPCV (N=8, k=2), significance, costs and backtest are frozen. See ../README.md section 2 and ../CONTEXT.md section 6. Any change is logged in ../DECISION_LOG.MD as a new trial first.

## Phase 2 is done when
- 07_phase2_qa.py passes 24/24 (needs labels; FinBERT macro F1 >= ~0.65 or a second scorer queued).
- Decision Log, README and these docs are updated.
- dev is merged to main and tagged phase2-complete.

## Top risks
- Low power (short, regime-specific sample): report nulls honestly.
- Noisy ticker tagging in FNSPID: many headlines are about other companies or market wrap-ups.
- Data files are git-ignored: back them up outside git.
- Time: the dashboard is capped at about 10% of project time and is cut first.
