# Blends of the long-history strategies

Daily walk-forward returns (base costs), 2014-01-02 to 2026-10-16, 3217 days, 25 strategies (16 long-only, 9 long-short; `deep_arb_long` left out).

| Portfolio | Sharpe | CAGR | Max DD | Alpha/yr vs SPY (t) | Beta |
|---|---|---|---|---|---|
| SPY buy and hold | 0.83 | +13.6% | -33.7% | – | 1.00 |
| best single (profitability_long) | 0.75 | +12.2% | -33.7% | -0.6% (-0.36) | 0.96 |
| equal: all long-only | 0.72 | +10.4% | -36.0% | -1.1% (-0.81) | 0.85 |
| equal: all long-short | -0.12 | -0.5% | -10.4% | +0.2% (0.20) | -0.04 |
| equal: long-only + long-short | 0.71 | +6.7% | -24.3% | -0.6% (-0.65) | 0.53 |
| inverse-vol: long-only | 0.72 | +9.8% | -34.0% | -1.1% (-0.80) | 0.80 |
| inverse-vol: long-short | -0.08 | -0.3% | -9.7% | +0.3% (0.35) | -0.04 |
| inverse-vol: all | 0.70 | +4.3% | -14.7% | -0.2% (-0.24) | 0.32 |
| long-only 70% + long-short 30% | 0.72 | +7.3% | -26.4% | -0.7% (-0.69) | 0.58 |

Average pairwise correlation: long-only 0.72, long-short 0.02, long-only with long-short -0.04.
