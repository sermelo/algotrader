# Live-tradable rule: the first decision of the day where the best stock's predicted edge reaches a threshold

One trade a day, cost 8 bp, out-of-sample 2013-2026 (3,330 days).

## Hold 90: all years, by threshold

| Threshold | Trades a year | Average decision (minutes after 09:30) | Gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|---|
| 0 bp | 247 | 55 | +6.1 bp | -1.9 bp | -0.5 | 49% |
| 2 bp | 191 | 100 | +8.7 bp | +0.7 bp | +0.2 | 51% |
| 4 bp | 118 | 125 | +12.0 bp | +4.0 bp | +0.7 | 52% |
| 6 bp | 67 | 137 | +21.8 bp | +13.8 bp | +1.8 | 53% |
| 8 bp | 41 | 146 | +24.6 bp | +16.6 bp | +1.7 | 54% |
| 10 bp | 28 | 158 | +38.1 bp | +30.1 bp | +2.6 | 58% |
| 15 bp | 15 | 174 | +70.9 bp | +62.9 bp | +3.7 | 65% |
| 20 bp | 7 | 179 | +114.6 bp | +106.6 bp | +3.0 | 72% |

### Hold 90: threshold chosen on earlier years only

| Year | Threshold chosen | Trades | Net per trade |
|---|---|---|---|
| 2014 | 8 bp | 64 | +29.5 bp |
| 2015 | 6 bp | 107 | +6.7 bp |
| 2016 | 6 bp | 123 | +16.4 bp |
| 2017 | 8 bp | 12 | +41.6 bp |
| 2018 | 8 bp | 43 | -50.9 bp |
| 2019 | 10 bp | 1 | -122.3 bp |
| 2020 | 6 bp | 82 | +1.8 bp |
| 2021 | 4 bp | 106 | -31.6 bp |
| 2022 | 15 bp | 22 | -18.4 bp |
| 2023 | 15 bp | 1 | +278.7 bp |
| 2024 | 15 bp | 0 | +nan bp |
| 2025 | 15 bp | 4 | -29.9 bp |
| 2026 | 6 bp | 8 | -107.3 bp |

All 13 years together: 573 trades, -2.6 bp net per trade.

## Hold 240: all years, by threshold

| Threshold | Trades a year | Average decision (minutes after 09:30) | Gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|---|
| 0 bp | 251 | 40 | +0.7 bp | -7.3 bp | -1.8 | 49% |
| 2 bp | 231 | 79 | -2.6 bp | -10.6 bp | -2.1 | 48% |
| 4 bp | 203 | 98 | -2.5 bp | -10.5 bp | -1.8 | 49% |
| 6 bp | 170 | 114 | -0.6 bp | -8.6 bp | -1.4 | 51% |
| 8 bp | 130 | 123 | +5.0 bp | -3.0 bp | -0.4 | 52% |
| 10 bp | 99 | 131 | +6.5 bp | -1.5 bp | -0.2 | 51% |
| 15 bp | 52 | 144 | +13.7 bp | +5.7 bp | +0.5 | 54% |
| 20 bp | 29 | 157 | +8.0 bp | +0.0 bp | +0.0 | 53% |

### Hold 240: threshold chosen on earlier years only

| Year | Threshold chosen | Trades | Net per trade |
|---|---|---|---|
| 2014 | 6 bp | 171 | -1.0 bp |
| 2015 | 20 bp | 58 | +13.4 bp |
| 2016 | 15 bp | 60 | -3.4 bp |
| 2017 | 20 bp | 14 | +132.1 bp |
| 2018 | 20 bp | 51 | -89.5 bp |
| 2019 | 15 bp | 52 | +83.8 bp |
| 2020 | 15 bp | 106 | -58.3 bp |
| 2021 | 15 bp | 54 | +24.5 bp |
| 2022 | 15 bp | 35 | +10.0 bp |
| 2023 | 15 bp | 6 | -106.1 bp |
| 2024 | 15 bp | 3 | +52.3 bp |
| 2025 | 15 bp | 12 | -35.1 bp |
| 2026 | 15 bp | 6 | +168.6 bp |

All 13 years together: 628 trades, -3.8 bp net per trade.

## Hold _close: all years, by threshold

| Threshold | Trades a year | Average decision (minutes after 09:30) | Gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|---|
| 0 bp | 252 | 31 | +10.7 bp | +2.7 bp | +0.5 | 50% |
| 2 bp | 245 | 44 | +10.4 bp | +2.4 bp | +0.5 | 50% |
| 4 bp | 217 | 71 | +8.6 bp | +0.6 bp | +0.1 | 50% |
| 6 bp | 180 | 89 | +9.8 bp | +1.8 bp | +0.2 | 49% |
| 8 bp | 143 | 98 | +10.8 bp | +2.8 bp | +0.3 | 50% |
| 10 bp | 114 | 108 | +11.4 bp | +3.4 bp | +0.3 | 49% |
| 15 bp | 60 | 120 | +11.4 bp | +3.4 bp | +0.3 | 51% |
| 20 bp | 35 | 137 | +29.6 bp | +21.6 bp | +1.2 | 53% |

### Hold _close: threshold chosen on earlier years only

| Year | Threshold chosen | Trades | Net per trade |
|---|---|---|---|
| 2014 | 20 bp | 23 | +67.2 bp |
| 2015 | 20 bp | 65 | +18.8 bp |
| 2016 | 0 bp | 252 | +26.7 bp |
| 2017 | 2 bp | 248 | +19.2 bp |
| 2018 | 2 bp | 250 | -27.1 bp |
| 2019 | 2 bp | 245 | +24.2 bp |
| 2020 | 0 bp | 253 | -20.6 bp |
| 2021 | 2 bp | 241 | -30.2 bp |
| 2022 | 20 bp | 29 | +30.2 bp |
| 2023 | 20 bp | 10 | -100.5 bp |
| 2024 | 20 bp | 10 | +171.9 bp |
| 2025 | 20 bp | 11 | -124.8 bp |
| 2026 | 20 bp | 5 | +201.4 bp |

All 13 years together: 1642 trades, +1.3 bp net per trade.

