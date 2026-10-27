# PLAN

Last updated: 2026-10-27

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
- Fallback: no calendar deadline (logged 2026-10-20); LSTM F0/F1 are committed to be completed. If LSTM is ever dropped, H1 needs 2 of 3 families, and the decision must be logged before any Phase 4 significance results are viewed.

## Scope in numbers
- 30 tickers, 1,258 trading days, 141,667 articles (98.2% FNSPID), 9 FRED series.
- Model rows: 35,790 (1,193 dates, 2019-03-29 to 2023-12-21), about 239 weekly cross-sections.
- Simulated power: 80% power needs a true IC gain of about 0.015-0.032, so a null H1 is plausible.
- FinBERT validation on 200 hand labels: macro F1 0.516 (95% CI 0.44-0.59), below the ~0.65 target. Second scorer finbert-tone (F1 0.581) queued as a robustness variant; ProsusAI/finbert stays primary.
- 9 core runs (Ridge, Random Forest, XGBoost, LSTM across tiers; see ../README.md section 2.5) plus optional TFT.

## Phases
| Phase | Focus | Status |
|---|---|---|
| 1 | Data collection and preprocessing | Done (QA 31/31) |
| 2 | Feature engineering | Done (QA 24/24); merged to main, tag phase2-complete |
| 3 | Models, MLflow tracking | In progress (3.0 and 3.1 done, 3.2 next) |
| 4 | Evaluation: CPCV, IC, significance, SHAP | Planned |
| 5 | Backtest vs equal-weight-30, F0, random, reversal | Planned |
| 6 | Read-only dashboard (FastAPI + React, local) | Planned, built last, first to cut |
| 7 | Cloud deployment | Cut |

## Phase 3 subphases
| Sub | Focus | Status |
|---|---|---|
| 3.0 | Close-out: data backup, requirements.txt verified, dev merged to main, docs | Done (2026-10-26) |
| 3.1 | CPCV splitter (N=8, k=2, purge 5, embargo 20), 7 paths, label-overlap unit tests | Done (2026-10-27, 7676aac; models/cpcv.py, tests/test_cpcv.py, 8 tests pass) |
| 3.2 | Fold-wise transformer (macro z-score, clip, 8 interactions) and shifted-fold test | Next |
| 3.3 | MLflow setup (SQLite, tags, git hash, checksums), seeds and splits fixed | Planned |
| 3.4 | Ridge F0/F1, XGBoost F0/F1/F2, Random Forest F0/F1 | Planned |
| 3.5 | LSTM F0/F1 | Planned |
| 3.6 | finbert-tone features and XGBoost robustness runs | Planned |
| 3.7 | Weekly rank IC, CSV export to results/, Phase 3 QA gate and report | Planned |

## Frozen design
Hypotheses, features, targets, CPCV (N=8, k=2), significance, costs and backtest are frozen. See ../README.md section 2 and CONTEXT.md section 6. Any change is logged in DECISION_LOG.MD as a new trial first.

## Phase 2 exit criteria
- [x] 07_phase2_qa.py passes 24/24 (FinBERT macro F1 >= ~0.65 or a second scorer queued; second scorer queued).
- [x] Decision Log and README updated.
- [x] dev merged to main and tagged phase2-complete (tag on c80280a; requirements.txt merged later via PR #1, 97729b7).

## Top risks
- Low power (short, regime-specific sample): report nulls honestly.
- Sentiment measurement quality: FinBERT validation F1 is below target and many headlines are off-topic (34% of the labeled sample). A null H1 has two explanations; the finbert-tone robustness run helps separate them.
- Noisy ticker tagging in FNSPID: many headlines are about other companies or market wrap-ups.
- Data files are git-ignored: backed up outside git (confirmed 2026-10-26); keep checksums in results/phase2_snapshot.json.
- Time: the dashboard is capped at about 10% of project time and is cut first.
