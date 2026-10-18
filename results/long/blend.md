# Blends of the long-history strategies

Daily walk-forward returns (base costs), 2014-01-02 to 2026-10-16, 3217 days, 25 strategies (16 long-only, 9 long-short; `deep_arb_long` left out).

| Portfolio | Sharpe | CAGR | Max DD | Alpha/yr vs SPY (t) | Beta |
|---|---|---|---|---|---|
| SPY buy and hold | 0.83 | +13.6% | -33.7% | – | 1.00 |
| best single (op_profitability_long) | 0.75 | +12.1% | -38.5% | -0.5% (-0.27) | 0.94 |
| equal: all long-only | 0.71 | +10.2% | -35.9% | -1.3% (-0.94) | 0.86 |
| equal: all long-short | -0.13 | -0.5% | -10.5% | +0.2% (0.21) | -0.05 |
| equal: long-only + long-short | 0.70 | +6.6% | -24.2% | -0.7% (-0.76) | 0.53 |
| inverse-vol: long-only | 0.70 | +9.7% | -33.9% | -1.3% (-0.94) | 0.80 |
| inverse-vol: long-short | -0.13 | -0.4% | -7.4% | +0.2% (0.23) | -0.04 |
| inverse-vol: all | 0.67 | +4.1% | -14.4% | -0.4% (-0.46) | 0.32 |
| long-only 70% + long-short 30% | 0.70 | +7.2% | -26.2% | -0.8% (-0.81) | 0.59 |

Average pairwise correlation: long-only 0.73, long-short 0.02, long-only with long-short -0.04.
