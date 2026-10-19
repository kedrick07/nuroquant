# BRAND

Last updated: 2026-10-19

This file covers both brand and voice.

## Name
NuroQuant (capital N, capital Q). One-line description: a research project that ranks US large-cap stocks weekly and tests whether FinBERT headline sentiment improves the ranking.

## Audience
- Primary: supervisor (Assoc. Prof. Dr. Azreen bin Azman) and examiners. They want rigor and honest limits.
- Secondary: thesis readers, GitHub visitors and internship recruiters. They want clean engineering and clear thinking.

## Personality
Rigorous, modest, transparent, practical. Claims are pre-registered; nulls are reported as findings.

## Voice
- Plain English, short sentences, active voice.
- State the result first, then the caveat.
- Scope every claim: this universe, this period (30 large caps, 2019-2023).
- Say headline sentiment from FNSPID, not news sentiment in general.
- Give numbers with units and intervals, never adjectives alone.
- Keep the same tone in README, thesis, poster, dashboard and commit messages.

## Preferred terms
| Use | Instead of |
|---|---|
| ranks stocks | predicts stock prices |
| weekly rank IC | accuracy (alone) |
| improves ranking by X IC (95% CI a to b) | significantly better (without a test) |
| in this universe and period | in the market |
| not supported by this sample | failed, did not work |

## Banned words and claims
beats the market, guaranteed, risk-free, holy grail, game-changing, revolutionary, AI-powered trading signals, proves, no survivorship bias, buy/sell/hold recommendations, alpha (unless measured and defined).

## Required disclaimer
Research project. Not financial advice. Results apply only to the 30-stock universe over 2019-2023.

## Formatting
- Dates ISO (YYYY-MM-DD). Tickers in capitals. Code and file names in backticks.
- Hypotheses are written H1 and H2; tiers F0, F1, F2.
- Commit messages: imperative and specific, for example: Add Phase 2 QA gate.
