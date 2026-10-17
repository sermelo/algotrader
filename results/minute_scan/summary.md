# Minute-feature scans (2008-2026, S&P 500 members, decisions at 10:00, 10:30 ... 15:00)

## Where are the big moves? Typical move (mean absolute return) as a multiple of a 0.14% cost, lowest to highest tenth of each condition

### Horizon 30 minutes

| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |
|---|---|---|---|---|
| dvol20 | 1.1x | 1.8x | 4.0x (0.0457) | 39% |
| vol30 | 0.9x | 1.8x | 5.0x (22.3) | 42% |
| range_bp | 1.2x | 1.7x | 5.0x (23.8) | 42% |
| rvol30 | 1.5x | 2.0x | 3.3x (2.72) | 37% |
| rvol_day | 2.6x | 1.8x | 2.6x (1.32) | 33% |
| abs_gap | 1.6x | 1.8x | 3.8x (0.0311) | 38% |
| abs_ret_open | 1.6x | 1.8x | 3.8x (0.036) | 38% |
| abs_rel30 | 1.5x | 1.7x | 3.8x (0.00899) | 39% |
| price | 3.2x | 1.8x | 1.9x (413) | 31% |
| logdv20 | 2.1x | 2.0x | 2.1x (20.8) | 30% |

### Horizon 90 minutes

| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |
|---|---|---|---|---|
| dvol20 | 1.7x | 2.9x | 6.6x (0.0457) | 44% |
| vol30 | 1.5x | 2.9x | 7.9x (22.3) | 45% |
| range_bp | 2.1x | 2.8x | 8.0x (23.8) | 45% |
| rvol30 | 2.6x | 3.4x | 5.1x (2.72) | 41% |
| rvol_day | 3.9x | 3.0x | 4.4x (1.32) | 39% |
| abs_gap | 2.6x | 3.0x | 6.3x (0.0311) | 43% |
| abs_ret_open | 2.6x | 3.0x | 6.3x (0.036) | 43% |
| abs_rel30 | 2.5x | 2.9x | 6.1x (0.00899) | 43% |
| price | 5.3x | 2.9x | 3.2x (413) | 38% |
| logdv20 | 3.6x | 3.3x | 3.4x (20.8) | 38% |

### Horizon 240 minutes

| Condition | Lowest tenth | Median tenth | Highest tenth | Highest tenth: share rising more than the cost |
|---|---|---|---|---|
| dvol20 | 2.4x | 4.0x | 9.4x (0.0457) | 46% |
| vol30 | 2.0x | 4.1x | 11.1x (22.3) | 47% |
| range_bp | 2.8x | 3.9x | 11.2x (23.8) | 47% |
| rvol30 | 3.6x | 4.7x | 6.8x (2.72) | 44% |
| rvol_day | 5.3x | 4.2x | 5.4x (1.32) | 41% |
| abs_gap | 3.5x | 4.1x | 9.0x (0.0311) | 46% |
| abs_ret_open | 3.6x | 4.2x | 8.6x (0.036) | 45% |
| abs_rel30 | 3.4x | 4.0x | 8.7x (0.00899) | 46% |
| price | 7.4x | 4.0x | 4.4x (413) | 41% |
| logdv20 | 5.0x | 4.6x | 4.7x (20.8) | 41% |

## Time of day (decision minute after 09:30): mean return and typical move

| Minute | Horizon 30: mean | Horizon 30: typical move | Horizon 90: mean | Horizon 90: typical move | To the close: mean |
|---|---|---|---|---|---|
| 30 | +0.004% | 0.438% | +0.002% | 0.648% | +0.025% |
| 60 | -0.003% | 0.362% | +0.003% | 0.561% | +0.022% |
| 90 | +0.001% | 0.320% | +0.006% | 0.501% | +0.024% |
| 120 | +0.005% | 0.283% | +0.009% | 0.447% | +0.024% |
| 150 | -0.000% | 0.259% | +0.004% | 0.423% | +0.019% |
| 180 | +0.004% | 0.240% | +0.002% | 0.415% | +0.019% |
| 210 | +0.001% | 0.237% | -0.000% | 0.420% | +0.014% |
| 240 | -0.002% | 0.237% | +0.011% | 0.422% | +0.015% |
| 270 | +0.001% | 0.252% | +0.018% | 0.436% | +0.016% |
| 300 | +0.012% | 0.243% | +0.016% | 0.473% | +0.016% |
| 330 | +0.005% | 0.252% | +0.004% | 0.397% | +0.004% |

## Which features rank stocks by later return? (114 feature x horizon tests; a |t| above about 3.3 is needed for 5% after testing this many)

IC = average rank correlation between the feature and the forward return across stocks at each decision; t over the days; years positive = calendar years with a positive average IC; excess returns in basis points against the average stock of the same decision (top/bottom tenth, and the single top/bottom stock).

| Feature | Horizon | IC | t | Years positive | Top tenth excess | Top stock excess | Top stock raw return | Bottom stock excess |
|---|---|---|---|---|---|---|---|---|
| ret5 | 10 | -0.0275 | -35.6 | 0/19 | -0.8 bp | -2.2 bp | -2.2 bp | +2.7 bp |
| range_pos | 10 | -0.0207 | -27.3 | 3/19 | -0.7 bp | -1.1 bp | -0.1 bp | +0.9 bp |
| rel30 | 10 | -0.0207 | -26.7 | 1/19 | -0.6 bp | -2.4 bp | -2.4 bp | +2.4 bp |
| ret30 | 10 | -0.0207 | -26.6 | 1/19 | -0.6 bp | -2.4 bp | -2.3 bp | +2.5 bp |
| ret15 | 10 | -0.0211 | -26.5 | 1/19 | -0.5 bp | -2.3 bp | -2.3 bp | +2.2 bp |
| vwap_dist | 10 | -0.0210 | -25.6 | 2/19 | -0.7 bp | -2.2 bp | -2.1 bp | +1.6 bp |
| ret5 | 20 | -0.0186 | -25.4 | 0/19 | -0.6 bp | -1.3 bp | -1.2 bp | +2.4 bp |
| ret60 | 10 | -0.0174 | -24.0 | 2/19 | -0.6 bp | -2.2 bp | -2.1 bp | +1.8 bp |
| rel30 | _close | -0.0164 | -22.2 | 0/19 | -0.7 bp | -1.9 bp | -0.1 bp | +5.4 bp |
| ret30 | _close | -0.0163 | -22.2 | 0/19 | -0.7 bp | -1.9 bp | -0.1 bp | +5.7 bp |
| ret30 | 90 | -0.0157 | -21.6 | 0/19 | -0.6 bp | -2.4 bp | -1.7 bp | +5.0 bp |
| rel30 | 90 | -0.0157 | -21.6 | 0/19 | -0.6 bp | -2.4 bp | -1.7 bp | +4.9 bp |
| ret30 | 240 | -0.0157 | -21.4 | 0/19 | -0.6 bp | -1.9 bp | -0.4 bp | +5.4 bp |
| rel30 | 240 | -0.0157 | -21.4 | 0/19 | -0.6 bp | -1.9 bp | -0.4 bp | +5.1 bp |
| ret5 | 30 | -0.0147 | -20.4 | 1/19 | -0.5 bp | -1.1 bp | -0.8 bp | +2.2 bp |
| ret30 | 20 | -0.0150 | -19.8 | 0/19 | -0.5 bp | -2.6 bp | -2.5 bp | +3.1 bp |
| rel30 | 20 | -0.0150 | -19.7 | 0/19 | -0.5 bp | -2.6 bp | -2.5 bp | +3.0 bp |
| ret15 | 20 | -0.0151 | -19.2 | 1/19 | -0.5 bp | -2.4 bp | -2.3 bp | +2.0 bp |
| ret60 | _close | -0.0164 | -19.2 | 0/19 | -0.8 bp | -2.0 bp | -0.3 bp | +5.4 bp |
| range_pos | _close | -0.0269 | -18.9 | 0/19 | -1.6 bp | -3.4 bp | +7.8 bp | +1.1 bp |
| ret30 | 30 | -0.0141 | -18.9 | 0/19 | -0.5 bp | -2.9 bp | -2.6 bp | +4.0 bp |
| rel30 | 30 | -0.0141 | -18.9 | 0/19 | -0.5 bp | -2.9 bp | -2.6 bp | +3.8 bp |
| ret15 | 90 | -0.0137 | -18.9 | 0/19 | -0.5 bp | -2.6 bp | -1.9 bp | +3.2 bp |
| ret60 | 240 | -0.0158 | -18.4 | 0/19 | -0.8 bp | -1.8 bp | -0.1 bp | +5.2 bp |
| ret15 | _close | -0.0134 | -18.2 | 0/19 | -0.4 bp | -2.0 bp | -0.2 bp | +3.8 bp |
