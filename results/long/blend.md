# Blends of the long-history strategies

Daily walk-forward returns (base costs), 2014-01-02 to 2026-11-04, 3230 days, 25 strategies (16 long-only, 9 long-short; `deep_arb_long` left out).

| Portfolio | Sharpe | CAGR | Max DD | Alpha/yr vs SPY (t) | Beta |
|---|---|---|---|---|---|
| SPY buy and hold | 0.83 | +13.6% | -33.7% | – | 1.00 |
| best single (op_profitability_long) | 0.80 | +13.1% | -38.0% | +0.4% (0.21) | 0.93 |
| equal: all long-only | 0.75 | +10.9% | -36.1% | -0.7% (-0.54) | 0.86 |
| equal: all long-short | -0.20 | -0.8% | -13.3% | -0.1% (-0.06) | -0.05 |
| equal: long-only + long-short | 0.73 | +6.9% | -24.4% | -0.5% (-0.50) | 0.53 |
| inverse-vol: long-only | 0.75 | +10.4% | -34.2% | -0.8% (-0.56) | 0.80 |
| inverse-vol: long-short | -0.14 | -0.4% | -8.5% | +0.1% (0.12) | -0.03 |
| inverse-vol: all | 0.72 | +4.5% | -14.7% | -0.2% (-0.18) | 0.32 |
| long-only 70% + long-short 30% | 0.74 | +7.6% | -26.4% | -0.5% (-0.52) | 0.59 |

Average pairwise correlation: long-only 0.72, long-short 0.02, long-only with long-short -0.04.
