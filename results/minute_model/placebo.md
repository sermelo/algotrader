# Placebo and ablation tests of the best-of-day result (hold 90 minutes, cost 0.08%)

| Variant | Mean IC | Trades a year | Gross per trade | Net per trade | Net t | Winning trades |
|---|---|---|---|---|---|---|
| real model (all features) | +0.0181 | 247 | +47.3 bp | +39.3 bp | +11.7 | 61% |
| shuffled targets (placebo) | +0.0067 | 250 | -4.6 bp | -12.6 bp | -5.8 | 43% |
| identity features only (price, dollar volume, minute) | +0.0072 | 243 | +2.6 bp | -5.4 bp | -2.9 | 45% |
| without price and dollar volume | +0.0184 | 247 | +49.5 bp | +41.5 bp | +12.7 | 62% |
| event features only (gap, volume surge, move since open, minute) | +0.0141 | 252 | +33.9 bp | +25.9 bp | +8.7 | 56% |
| short-term features only (returns, VWAP, range, SPY) | +0.0194 | 249 | +34.7 bp | +26.7 bp | +7.8 | 57% |
| random pick among the volatile tenth | +0.0000 | 252 | -0.2 bp | -8.2 bp | -4.7 | 45% |
