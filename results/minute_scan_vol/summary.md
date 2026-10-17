# Minute-feature scans (2008-2026, S&P 500 members, decisions at 10:00, 10:30 ... 15:00; only the top 10% of stocks by 30-minute volatility at each decision)

## Where are the big moves? Typical move (mean absolute return) as a multiple of a 0.14% cost, lowest to highest tenth of each condition

### Horizon 30 minutes

| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |
|---|---|---|---|---|
| dvol20 | 2.2x | 2.5x | 4.7x (0.049) | 41% |
| vol30 | 0.5x | 1.7x | 5.4x (25.8) | 42% |
| range_bp | 2.4x | 2.0x | 5.5x (27.1) | 42% |
| rvol30 | 2.6x | 3.2x | 4.6x (3.41) | 40% |
| rvol_day | 4.5x | 3.2x | 3.8x (1.65) | 38% |
| abs_gap | 2.8x | 2.9x | 4.9x (0.0427) | 41% |
| abs_ret_open | 2.9x | 2.9x | 4.6x (0.0414) | 40% |
| abs_rel30 | 2.5x | 2.7x | 4.6x (0.0107) | 41% |
| price | 4.2x | 3.1x | 3.2x (432) | 38% |
| logdv20 | 4.1x | 3.3x | 3.8x (21.1) | 39% |

### Horizon 90 minutes

| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |
|---|---|---|---|---|
| dvol20 | 3.5x | 4.0x | 7.7x (0.049) | 44% |
| vol30 | 0.4x | 2.8x | 8.5x (25.8) | 45% |
| range_bp | 4.1x | 3.3x | 8.7x (27.1) | 45% |
| rvol30 | 4.5x | 5.3x | 7.0x (3.41) | 44% |
| rvol_day | 6.8x | 5.2x | 6.1x (1.65) | 43% |
| abs_gap | 4.4x | 4.6x | 7.9x (0.0427) | 44% |
| abs_ret_open | 4.6x | 4.7x | 7.5x (0.0414) | 44% |
| abs_rel30 | 4.2x | 4.4x | 7.4x (0.0107) | 44% |
| price | 6.9x | 4.9x | 5.1x (432) | 42% |
| logdv20 | 6.8x | 5.3x | 6.0x (21.1) | 43% |

### Horizon 240 minutes

| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |
|---|---|---|---|---|
| dvol20 | 4.5x | 5.2x | 10.6x (0.049) | 46% |
| vol30 | 0.5x | 3.4x | 11.7x (25.8) | 47% |
| range_bp | 5.6x | 4.2x | 11.9x (27.1) | 47% |
| rvol30 | 6.2x | 7.2x | 9.3x (3.41) | 45% |
| rvol_day | 9.2x | 7.2x | 7.7x (1.65) | 44% |
| abs_gap | 5.9x | 6.2x | 10.9x (0.0427) | 46% |
| abs_ret_open | 6.2x | 6.4x | 10.1x (0.0414) | 45% |
| abs_rel30 | 5.7x | 6.0x | 10.3x (0.0107) | 46% |
| price | 9.6x | 6.6x | 6.8x (432) | 44% |
| logdv20 | 9.4x | 7.2x | 8.2x (21.1) | 45% |

## Time of day (decision minute after 09:30): mean return and typical move

| Minute | Horizon 30: mean | Horizon 30: typical move | Horizon 90: mean | Horizon 90: typical move | To the close: mean |
|---|---|---|---|---|---|
| 30 | -0.000% | 0.753% | -0.009% | 1.105% | +0.027% |
| 60 | -0.005% | 0.645% | +0.003% | 0.985% | +0.032% |
| 90 | -0.001% | 0.565% | +0.011% | 0.870% | +0.037% |
| 120 | +0.012% | 0.496% | +0.018% | 0.770% | +0.043% |
| 150 | +0.000% | 0.450% | +0.002% | 0.726% | +0.029% |
| 180 | +0.004% | 0.415% | -0.003% | 0.709% | +0.032% |
| 210 | +0.000% | 0.411% | -0.002% | 0.712% | +0.029% |
| 240 | -0.005% | 0.410% | +0.015% | 0.713% | +0.031% |
| 270 | +0.003% | 0.425% | +0.024% | 0.723% | +0.034% |
| 300 | +0.019% | 0.411% | +0.034% | 0.757% | +0.034% |
| 330 | +0.004% | 0.421% | +0.013% | 0.625% | +0.013% |

## Which features rank stocks by later return? (138 feature x horizon tests; a |t| above about 3.3 is needed for 5% after testing this many)

IC = average rank correlation between the feature and the forward return across stocks at each decision; t over the days; years positive = calendar years with a positive average IC; excess returns in basis points against the average stock of the same decision (top/bottom tenth, and the single top/bottom stock).

| Feature | Horizon | IC | t | Years positive | Top tenth excess | Top stock excess | Top stock raw return | Bottom stock excess |
|---|---|---|---|---|---|---|---|---|
| z5 | 10 | -0.0349 | -37.2 | 0/19 | -1.4 bp | -1.8 bp | -1.8 bp | +2.2 bp |
| ret5 | 10 | -0.0355 | -37.1 | 0/19 | -1.5 bp | -2.1 bp | -2.0 bp | +2.7 bp |
| z30 | 10 | -0.0288 | -28.8 | 1/19 | -1.2 bp | -2.1 bp | -2.1 bp | +2.1 bp |
| ret30 | 10 | -0.0294 | -28.8 | 1/19 | -1.2 bp | -2.1 bp | -2.1 bp | +2.3 bp |
| rel30 | 10 | -0.0293 | -28.8 | 1/19 | -1.2 bp | -2.1 bp | -2.1 bp | +2.3 bp |
| ret15 | 10 | -0.0276 | -28.4 | 2/19 | -1.1 bp | -1.9 bp | -1.9 bp | +2.0 bp |
| ret5 | 20 | -0.0253 | -27.7 | 0/19 | -1.2 bp | -1.3 bp | -1.1 bp | +2.4 bp |
| z5 | 20 | -0.0246 | -27.5 | 0/19 | -1.1 bp | -1.6 bp | -1.4 bp | +1.7 bp |
| ret60 | 10 | -0.0268 | -24.6 | 2/19 | -1.2 bp | -2.2 bp | -2.1 bp | +1.8 bp |
| range_pos | 10 | -0.0234 | -24.5 | 3/19 | -1.1 bp | -1.9 bp | -1.6 bp | +1.5 bp |
| ret30 | 20 | -0.0245 | -24.4 | 1/19 | -1.3 bp | -2.5 bp | -2.3 bp | +3.1 bp |
| rel30 | 20 | -0.0245 | -24.3 | 1/19 | -1.3 bp | -2.5 bp | -2.3 bp | +2.8 bp |
| vwap_dist | 10 | -0.0243 | -24.2 | 3/19 | -1.3 bp | -2.2 bp | -2.2 bp | +1.5 bp |
| z30 | 20 | -0.0237 | -24.0 | 1/19 | -1.2 bp | -2.1 bp | -2.0 bp | +2.1 bp |
| ret30 | 90 | -0.0236 | -23.6 | 0/19 | -1.5 bp | -2.5 bp | -1.4 bp | +5.1 bp |
| ret30 | 30 | -0.0237 | -23.5 | 0/19 | -1.5 bp | -2.7 bp | -2.4 bp | +4.0 bp |
| rel30 | 90 | -0.0235 | -23.5 | 0/19 | -1.5 bp | -2.4 bp | -1.3 bp | +4.8 bp |
| ret5 | 30 | -0.0213 | -23.5 | 2/19 | -1.2 bp | -1.1 bp | -0.8 bp | +2.2 bp |
| rel30 | 30 | -0.0236 | -23.5 | 0/19 | -1.4 bp | -2.7 bp | -2.4 bp | +3.7 bp |
| ret15 | 20 | -0.0222 | -23.5 | 1/19 | -1.2 bp | -1.9 bp | -1.8 bp | +2.1 bp |
| z30 | 90 | -0.0228 | -23.3 | 0/19 | -1.5 bp | -2.2 bp | -1.0 bp | +2.6 bp |
| z5 | 30 | -0.0207 | -23.2 | 2/19 | -1.1 bp | -1.6 bp | -1.3 bp | +1.5 bp |
| z30 | 30 | -0.0226 | -22.9 | 0/19 | -1.3 bp | -2.4 bp | -2.1 bp | +2.5 bp |
| ret30 | _close | -0.0227 | -22.8 | 0/19 | -1.6 bp | -3.2 bp | -0.0 bp | +5.0 bp |
| rel30 | _close | -0.0226 | -22.7 | 0/19 | -1.6 bp | -3.1 bp | +0.0 bp | +4.4 bp |
