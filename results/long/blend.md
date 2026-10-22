# Blends of the long-history strategies

Daily walk-forward returns (base costs), 2014-01-02 to 2026-10-16, 3217 days, 25 strategies (16 long-only, 9 long-short; `deep_arb_long` left out).

| Portfolio | Sharpe | CAGR | Max DD | Alpha/yr vs SPY (t) | Beta |
|---|---|---|---|---|---|
| SPY buy and hold | 0.83 | +13.6% | -33.7% | – | 1.00 |
| best single (op_profitability_long) | 0.74 | +12.0% | -38.5% | -0.5% (-0.28) | 0.94 |
| equal: all long-only | 0.71 | +10.2% | -36.0% | -1.3% (-0.98) | 0.86 |
| equal: all long-short | -0.13 | -0.5% | -10.5% | +0.2% (0.21) | -0.05 |
| equal: long-only + long-short | 0.70 | +6.5% | -24.2% | -0.8% (-0.79) | 0.53 |
| inverse-vol: long-only | 0.70 | +9.6% | -34.0% | -1.4% (-0.99) | 0.80 |
| inverse-vol: long-short | -0.13 | -0.4% | -7.4% | +0.2% (0.23) | -0.04 |
| inverse-vol: all | 0.66 | +4.0% | -14.4% | -0.4% (-0.51) | 0.32 |
| long-only 70% + long-short 30% | 0.70 | +7.1% | -26.3% | -0.9% (-0.85) | 0.59 |

Average pairwise correlation: long-only 0.73, long-short 0.02, long-only with long-short -0.04.
