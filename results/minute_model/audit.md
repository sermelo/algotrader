# Audit of the model's best trade of the day

1,618,933 out-of-sample decisions in the volatile tenth, 3,330 days.

## Hold 90 minutes: the 3,330 best-of-day trades (no filter)

Mean gross +47.0 bp, median +38.3 bp, mean after cutting the biggest and smallest 1%: +45.4 bp, after cutting 5%: +43.7 bp. Trades gaining over 5%: 1.1%, losing over 5%: 0.4%; best +53.7%, worst -15.8%.

| | Price | Trades in last 30 min | 30-min volatility (bp/min) | log dollar volume | Bar range (bp) | Volume surge | Last 5-min move |
|---|---|---|---|---|---|---|---|
| Picked trades (median) | 56.4 | 30 | 18.7 | 19 | 19.6 | 2.63 | 0.00336 |
| Volatile tenth (median) | 54 | 30 | 10.8 | 19 | 9.59 | 0.981 | 0.00141 |

Ten biggest winners:

| Date | Time | Ticker | Price | Last 5-min move | Opening gap | Volume surge | Predicted | Realised |
|---|---|---|---|---|---|---|---|---|
| 2023-03-13 | 10:30 | FRC | 19.00 | -4.8% | -64.8% | 9.0x | +4 bp | +53.7% |
| 2020-03-18 | 14:30 | SYY | 26.66 | +2.2% | -4.1% | 2.9x | +26 bp | +17.1% |
| 2015-10-21 | 13:30 | ENDP | 51.56 | +0.7% | +1.0% | 6.1x | +25 bp | +12.2% |
| 2016-01-26 | 11:00 | FCX | 3.90 | -1.0% | +11.7% | 2.3x | +12 bp | +9.5% |
| 2021-10-14 | 10:00 | WBA | 45.56 | -0.1% | +0.4% | 9.8x | +2 bp | +9.2% |
| 2021-03-05 | 11:30 | ENPH | 130.08 | +0.0% | +2.5% | 3.1x | +25 bp | +9.1% |
| 2017-02-28 | 10:30 | ENDP | 13.19 | -0.7% | +5.3% | 4.3x | +5 bp | +9.0% |
| 2022-07-28 | 10:00 | ALGN | 257.31 | -0.7% | +3.1% | 6.8x | +7 bp | +8.9% |
| 2018-08-03 | 12:00 | DISH | 31.10 | -0.2% | +4.9% | 1.2x | +5 bp | +8.8% |
| 2017-11-01 | 10:00 | EVHC | 27.59 | -1.8% | -29.4% | 56.3x | +7 bp | +8.7% |

## Hold 240 minutes: the 3,330 best-of-day trades (no filter)

Mean gross +46.5 bp, median +34.8 bp, mean after cutting the biggest and smallest 1%: +45.2 bp, after cutting 5%: +43.8 bp. Trades gaining over 5%: 2.0%, losing over 5%: 0.6%; best +26.9%, worst -40.5%.

| | Price | Trades in last 30 min | 30-min volatility (bp/min) | log dollar volume | Bar range (bp) | Volume surge | Last 5-min move |
|---|---|---|---|---|---|---|---|
| Picked trades (median) | 51.7 | 30 | 17.6 | 19 | 18.8 | 2.39 | 0.00298 |
| Volatile tenth (median) | 54 | 30 | 10.8 | 19 | 9.59 | 0.981 | 0.00141 |

Ten biggest winners:

| Date | Time | Ticker | Price | Last 5-min move | Opening gap | Volume surge | Predicted | Realised |
|---|---|---|---|---|---|---|---|---|
| 2020-03-18 | 13:30 | HLT | 44.95 | -1.0% | -10.5% | 3.2x | +36 bp | +26.9% |
| 2020-03-10 | 11:30 | RCL | 42.33 | -0.6% | +6.1% | 4.8x | +36 bp | +20.3% |
| 2016-01-14 | 11:30 | WMB | 14.71 | -0.3% | -1.4% | 2.4x | +29 bp | +19.7% |
| 2025-04-09 | 11:00 | MU | 63.78 | -0.5% | +1.0% | 1.6x | +9 bp | +18.3% |
| 2015-02-19 | 10:00 | DNR | 7.96 | -1.0% | -3.6% | 2.7x | +5 bp | +15.4% |
| 2023-04-19 | 10:30 | FRC | 12.74 | -0.2% | +2.6% | 0.5x | +1 bp | +13.7% |
| 2015-08-06 | 11:00 | VIAB | 40.68 | -0.5% | -4.9% | 18.0x | +41 bp | +11.7% |
| 2024-04-12 | 12:30 | GL | 53.01 | -0.2% | +21.5% | 7.0x | +17 bp | +11.5% |
| 2015-10-08 | 13:00 | NFLX | 103.15 | +0.0% | -0.1% | 1.4x | +12 bp | +11.5% |
| 2019-01-16 | 12:00 | PCG | 6.36 | -1.5% | -6.7% | 2.0x | +23 bp | +10.9% |

## Best-of-day with the filter applied before the pick (cost 0.08%)

### Hold 90 minutes

| Filter | Candidates kept | Trades a year | Gross | Median gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|---|---|
| no filter | 100% | 247 | +47.3 bp | +38.3 bp | +39.3 bp | +11.7 | 61% |
| traded in 25+ of the last 30 minutes | 92% | 246 | +46.6 bp | +38.7 bp | +38.6 bp | +13.0 | 62% |
| price at least $50 | 53% | 234 | +43.6 bp | +35.9 bp | +35.6 bp | +14.1 | 62% |
| last 5-min move under 3% | 100% | 247 | +45.1 bp | +38.1 bp | +37.1 bp | +12.4 | 61% |
| opening gap under 5% | 95% | 245 | +45.0 bp | +37.6 bp | +37.0 bp | +13.8 | 61% |
| liquid (top half by dollar volume) and traded 25+ of 30 minutes | 47% | 229 | +46.9 bp | +37.2 bp | +38.9 bp | +13.7 | 63% |
| all of: liquid, 25+ of 30 minutes, price $50+, move under 3%, gap under 5% | 28% | 208 | +42.6 bp | +29.4 bp | +34.6 bp | +13.8 | 62% |

### Hold 240 minutes

| Filter | Candidates kept | Trades a year | Gross | Median gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|---|---|
| no filter | 100% | 251 | +46.6 bp | +34.8 bp | +38.6 bp | +10.3 | 58% |
| traded in 25+ of the last 30 minutes | 92% | 251 | +45.0 bp | +33.6 bp | +37.0 bp | +9.9 | 58% |
| price at least $50 | 53% | 249 | +44.0 bp | +35.4 bp | +36.0 bp | +11.1 | 59% |
| last 5-min move under 3% | 100% | 251 | +45.2 bp | +34.7 bp | +37.2 bp | +9.8 | 58% |
| opening gap under 5% | 95% | 251 | +45.3 bp | +34.8 bp | +37.3 bp | +11.6 | 58% |
| liquid (top half by dollar volume) and traded 25+ of 30 minutes | 47% | 248 | +43.1 bp | +34.6 bp | +35.1 bp | +10.1 | 58% |
| all of: liquid, 25+ of 30 minutes, price $50+, move under 3%, gap under 5% | 28% | 241 | +42.7 bp | +33.6 bp | +34.7 bp | +12.3 | 60% |

## Cost stress: net per trade after extra round-trip costs (hold 90 minutes)

| Extra cost | No filter | All filters |
|---|---|---|
| +0 bp | +39.3 bp (t +11.7) | +34.6 bp (t +13.8) |
| +5 bp | +34.3 bp (t +10.2) | +29.6 bp (t +11.8) |
| +10 bp | +29.3 bp (t +8.7) | +24.6 bp (t +9.8) |
| +20 bp | +19.3 bp (t +5.7) | +14.6 bp (t +5.8) |
| +40 bp | -0.7 bp (t -0.2) | -5.4 bp (t -2.2) |

## By market stress (hold 90, no filter)

Stress = the day's median 30-minute volatility in the volatile tenth, in quarters.

| Stress | Trades | Gross | Median gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|---|
| calm | 833 | +34.8 bp | +23.3 bp | +26.8 bp | +6.6 | 58% |
| normal | 832 | +40.3 bp | +35.5 bp | +32.3 bp | +6.3 | 60% |
| busy | 832 | +37.0 bp | +44.9 bp | +29.0 bp | +4.9 | 62% |
| stressed | 833 | +75.9 bp | +56.4 bp | +67.9 bp | +7.0 | 63% |

## Is the predicted edge calibrated? (hold 90, best trade of each day, by fifths of the prediction)

| Fifth | Predicted edge | Realised gross | Realised net | Net t |
|---|---|---|---|---|
| 1 | +0.9 bp | +39.9 bp | +31.9 bp | +8.8 |
| 2 | +2.3 bp | +42.3 bp | +34.3 bp | +6.8 |
| 3 | +3.8 bp | +49.1 bp | +41.1 bp | +4.1 |
| 4 | +5.7 bp | +38.2 bp | +30.2 bp | +3.9 |
| 5 | +13.1 bp | +65.7 bp | +57.7 bp | +6.7 |

## When the best trade happens (hold 90)

| Decision at | Share of days | Net per trade |
|---|---|---|
| 10:00 | 10% | +77.6 bp |
| 10:30 | 13% | +80.7 bp |
| 11:00 | 16% | +35.8 bp |
| 11:30 | 10% | +32.8 bp |
| 12:00 | 9% | +29.1 bp |
| 12:30 | 7% | +42.8 bp |
| 13:00 | 6% | +26.2 bp |
| 13:30 | 7% | +31.2 bp |
| 14:00 | 6% | +19.0 bp |
| 14:30 | 8% | +12.2 bp |
| 15:00 | 8% | +4.3 bp |

## By year (hold 90, no filter): net per trade

| Year | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| net bp | +33 | +47 | +53 | +57 | +29 | +48 | +28 | +51 | +43 | +40 | +44 | +6 | +35 | +5 |
