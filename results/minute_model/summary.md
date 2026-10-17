# Walk-forward gradient boosting on the volatile tenth of S&P 500 stocks

Universe: top 10% of the cross-section by 30-minute volatility at each decision; 1,618,933 out-of-sample decisions, 2013-2026; features: 27; one model per horizon, refitted every year on all earlier years.

## Rank correlation between prediction and realised return (IC), by year

| Year | 30 | 90 | 240 | _close |
|---|---|---|---|---|
| 2013 | +0.0272 | +0.0386 | +0.0496 | +0.0530 |
| 2014 | +0.0196 | +0.0253 | +0.0300 | +0.0246 |
| 2015 | +0.0135 | +0.0187 | +0.0282 | +0.0309 |
| 2016 | +0.0162 | +0.0154 | +0.0234 | +0.0291 |
| 2017 | +0.0284 | +0.0193 | +0.0303 | +0.0340 |
| 2018 | +0.0357 | +0.0281 | +0.0351 | +0.0383 |
| 2019 | +0.0380 | +0.0223 | +0.0226 | +0.0247 |
| 2020 | +0.0242 | +0.0172 | +0.0184 | +0.0187 |
| 2021 | +0.0177 | +0.0131 | +0.0203 | +0.0240 |
| 2022 | -0.0008 | +0.0108 | +0.0115 | +0.0109 |
| 2023 | +0.0268 | +0.0219 | +0.0382 | +0.0395 |
| 2024 | +0.0189 | +0.0165 | +0.0232 | +0.0266 |
| 2025 | +0.0140 | +0.0055 | +0.0047 | +0.0021 |
| 2026 | +0.0031 | +0.0008 | +0.0074 | +0.0170 |
| mean | +0.0202 | +0.0181 | +0.0245 | +0.0267 |

## The 12 pre-registered tests (costs: 1 pick 0.08%, 3 picks 0.14% each)

| Hold | Decision | Picks | Gross per trade | Excess over the volatile tenth | Net of cost | Net t | Winning trades | Years net > 0 |
|---|---|---|---|---|---|---|---|---|
| 240 | 10:00 | 1 | -0.5 bp | -0.9 bp (t -0.2) | -8.5 bp | -2.0 | 49% | 6/14 |
| 240 | 10:30 | 1 | +1.8 bp | +0.5 bp (t +0.1) | -6.2 bp | -1.4 | 49% | 5/14 |
| 240 | 11:00 | 1 | +2.9 bp | +0.2 bp (t +0.1) | -5.1 bp | -1.4 | 49% | 5/14 |
| 240 | 10:00 | 3 | +1.7 bp | +1.4 bp (t +0.6) | -12.3 bp | -4.5 | 46% | 0/14 |
| 240 | 10:30 | 3 | +4.5 bp | +3.3 bp (t +1.6) | -9.5 bp | -3.4 | 46% | 2/14 |
| 240 | 11:00 | 3 | +4.4 bp | +1.8 bp (t +1.0) | -9.6 bp | -3.9 | 46% | 0/14 |
| 90 | 10:00 | 1 | +4.8 bp | +5.3 bp (t +1.6) | -3.2 bp | -0.9 | 48% | 6/14 |
| 90 | 10:30 | 1 | +5.7 bp | +4.8 bp (t +1.6) | -2.3 bp | -0.7 | 48% | 6/14 |
| 90 | 11:00 | 1 | +2.9 bp | +0.8 bp (t +0.4) | -5.1 bp | -2.1 | 48% | 3/14 |
| 90 | 10:00 | 3 | +4.2 bp | +4.7 bp (t +2.9) | -9.8 bp | -4.6 | 45% | 2/14 |
| 90 | 10:30 | 3 | +3.3 bp | +2.4 bp (t +1.7) | -10.7 bp | -5.7 | 43% | 0/14 |
| 90 | 11:00 | 3 | +3.3 bp | +1.1 bp (t +0.9) | -10.7 bp | -6.6 | 43% | 1/14 |

Primary tests with net t above 3 and a positive result in most years: **0 of 12**.

## All 88 (horizon x decision minute x picks) tests: the best

Luck alone would give a best t of about 3.0.

| Hold | Decision | Picks | Gross per trade | Excess over the volatile tenth | Net of cost | Net t | Winning trades | Years net > 0 |
|---|---|---|---|---|---|---|---|---|
| _close | 10:30 | 1 | +13.3 bp | +11.4 bp (t +2.8) | +5.3 bp | +1.1 | 50% | 9/14 |
| _close | 11:30 | 1 | +11.2 bp | +9.0 bp (t +2.2) | +3.2 bp | +0.7 | 50% | 8/14 |
| _close | 10:00 | 1 | +10.1 bp | +8.9 bp (t +2.1) | +2.1 bp | +0.4 | 50% | 9/14 |
| 240 | 11:30 | 1 | +7.2 bp | +4.7 bp (t +1.3) | -0.8 bp | -0.2 | 50% | 8/14 |
| _close | 11:00 | 1 | +6.2 bp | +3.9 bp (t +1.1) | -1.8 bp | -0.4 | 50% | 8/14 |
| _close | 14:00 | 1 | +6.1 bp | +5.1 bp (t +2.1) | -1.9 bp | -0.7 | 48% | 6/14 |
| 90 | 10:30 | 1 | +5.7 bp | +4.8 bp (t +1.6) | -2.3 bp | -0.7 | 48% | 6/14 |
| 90 | 10:00 | 1 | +4.8 bp | +5.3 bp (t +1.6) | -3.2 bp | -0.9 | 48% | 6/14 |
| _close | 13:30 | 1 | +4.3 bp | +3.2 bp (t +1.2) | -3.7 bp | -1.2 | 48% | 6/14 |
| 240 | 14:30 | 1 | +4.9 bp | +4.3 bp (t +2.1) | -3.1 bp | -1.3 | 48% | 5/14 |

## The most confident trade of each day (one trade a day, any decision minute), by prediction threshold

### Hold 90: cost 0.08%

| Predicted edge at least | Trades a year | Gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|
| 0 bp | 247 | +47.3 bp | +39.3 bp | +11.7 | 61% |
| 5 bp | 89 | +54.2 bp | +46.2 bp | +7.5 | 61% |
| 10 bp | 28 | +81.6 bp | +73.6 bp | +5.8 | 66% |
| 15 bp | 15 | +103.1 bp | +95.1 bp | +5.1 | 72% |
| 20 bp | 7 | +123.5 bp | +115.5 bp | +3.4 | 75% |
| 30 bp | 1 | +82.5 bp | +74.5 bp | +1.1 | 67% |
| 50 bp | 0 | +nan bp | +nan bp | +nan | nan% |

### Hold 240: cost 0.08%

| Predicted edge at least | Trades a year | Gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|
| 0 bp | 251 | +46.6 bp | +38.6 bp | +10.3 | 58% |
| 5 bp | 187 | +48.9 bp | +40.9 bp | +8.7 | 58% |
| 10 bp | 99 | +59.4 bp | +51.4 bp | +7.6 | 59% |
| 15 bp | 52 | +79.4 bp | +71.4 bp | +6.6 | 61% |
| 20 bp | 29 | +84.7 bp | +76.7 bp | +4.6 | 62% |
| 30 bp | 7 | +155.4 bp | +147.4 bp | +3.2 | 59% |
| 50 bp | 0 | +227.5 bp | +219.5 bp | +1.9 | 80% |

### Hold _close: cost 0.08%

| Predicted edge at least | Trades a year | Gross | Net | Net t | Winning trades |
|---|---|---|---|---|---|
| 0 bp | 252 | +45.6 bp | +37.6 bp | +9.6 | 56% |
| 5 bp | 198 | +48.1 bp | +40.1 bp | +8.6 | 56% |
| 10 bp | 114 | +57.6 bp | +49.6 bp | +7.0 | 57% |
| 15 bp | 60 | +70.3 bp | +62.3 bp | +5.5 | 57% |
| 20 bp | 35 | +84.0 bp | +76.0 bp | +4.9 | 58% |
| 30 bp | 10 | +114.4 bp | +106.4 bp | +2.5 | 59% |
| 50 bp | 1 | +377.3 bp | +369.3 bp | +2.0 | 71% |

