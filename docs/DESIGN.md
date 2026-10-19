# DESIGN

Last updated: 2026-10-19

Status: proposed defaults for the Phase 6 dashboard. Confirm before building.

## Purpose and scope
A local, read-only viewer of frozen, precomputed results. Stack: PostgreSQL, FastAPI, React. No inference, no auth, no live data. Kedrick writes the FastAPI endpoints and schema; Lovable is for React scaffolding only. Built after the Phase 5 freeze and capped at about 10% of project time.

## Pages
1. Results: model x tier table of mean weekly rank IC, 95% CI, BH-adjusted p and a verdict badge (H1, H2).
2. Backtest: net equity curve (median across the 7 CPCV paths with a min-max band) vs equal-weight-30, F0, random and reversal baselines; cost sensitivity; drawdown.
3. Rankings: date picker and a table of rank, ticker and score for the 30 stocks. Ranks, not trade signals.
4. Explainability and Exposures: group SHAP shares, permutation importance, market, factor and sector exposures.

Every page shows a limitations banner at the top and a footer with MLflow run IDs and the data checksum.

## Color tokens
| Token | Hex | Use |
|---|---|---|
| bg | #F8FAFC | Page background |
| surface | #FFFFFF | Cards, tables |
| text | #0F172A | Body text |
| muted | #64748B | Secondary text |
| primary | #1D4ED8 | Links, selected state |
| warn-bg / warn-text | #FEF3C7 / #92400E | Limitations banner |
| long / short | #15803D / #B91C1C | Long and short legs only |
| F0 / F1 / F2 | #56B4E9 / #E69F00 / #009E73 | Tier series (color-blind safe) |
| baseline | #999999 | Equal-weight, random, reversal |

Significance is not shown in green or red. Use neutral badges: Supported, Not supported.

## Typography and layout
- Font: Inter (fallback system-ui). Numbers in tabular figures or a mono font.
- Sizes: 14px body, 12px table detail, 20px page title.
- Max content width 1200px. 8px spacing grid. Tables first, charts second.

## Components
- Metric table: sortable, sticky header, CI next to every point estimate.
- Verdict badge: Supported, Not supported or Exploratory.
- Line chart: time on x, tooltips with date and value, legend uses tier colors.
- Bar chart: horizontal bars for SHAP group shares.
- Banner: always visible, text from the limitations list in ../README.md section 2.1.3.
- Run footer: run IDs, git hash, data checksum.

## Rules
- Always show uncertainty (CI or range) with a point estimate.
- Number formats: IC 3 decimals, Sharpe 2 decimals, returns 1 decimal percent, dates ISO (YYYY-MM-DD).
- No buy, sell or hold wording. No profit claims. See BRAND.md.
- Contrast at least 4.5:1; never rely on color alone (add labels or patterns).
- Null results get the same visual weight as positive ones.
