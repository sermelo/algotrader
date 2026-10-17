# Simple stock-picking rules against the cost of trading (2008-2026)

6,264 tests (rule x decision minute x horizon). With that many tests the best |t| expected from pure luck is about 4.2: only a t clearly above that, in most years, is evidence.
Volatility filter on the 30-minute realised volatility: volatile half >= 5.7 bp a minute, most volatile tenth >= 13.8. 1 pick costs 0.08% (a whole 2,000-euro position), 3 picks 0.14% each (three slots), Tiered, 2 bp half-spread a side.

Rules with a positive result after costs: 88 of 6264; with net t above 4.2: 0; excess t above 4.2 or below -4.2 (a real stock-picking effect, whatever the cost): 125.

## Best by result after costs

| Pick | Score | Filter | k | Minute | Hold | Gross | Excess over filter | Excess t | Net | Net t | Winning days | Years net > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | ret30 | most volatile tenth | 1 | 300 | 90 | +13.2 bp | +9.1 bp | +3.0 | +5.2 bp | +1.4 | 48% | 9/19 |
| low | ret30 | most volatile tenth | 1 | 300 | _close | +13.2 bp | +9.1 bp | +3.0 | +5.2 bp | +1.4 | 48% | 9/19 |
| low | ret30 | most volatile tenth | 1 | 300 | 240 | +13.2 bp | +9.1 bp | +3.0 | +5.2 bp | +1.4 | 48% | 9/19 |
| low | prev_ret | most volatile tenth | 1 | 240 | _close | +13.6 bp | +11.7 bp | +3.2 | +5.6 bp | +1.2 | 48% | 10/19 |
| low | prev_ret | most volatile tenth | 1 | 240 | 240 | +13.6 bp | +11.7 bp | +3.2 | +5.6 bp | +1.2 | 48% | 10/19 |
| low | prev_ret | most volatile tenth | 1 | 300 | 240 | +12.1 bp | +8.2 bp | +2.8 | +4.1 bp | +1.1 | 47% | 8/19 |
| low | prev_ret | most volatile tenth | 1 | 300 | _close | +12.1 bp | +8.2 bp | +2.8 | +4.1 bp | +1.1 | 47% | 8/19 |
| low | prev_ret | most volatile tenth | 1 | 300 | 90 | +12.1 bp | +8.2 bp | +2.8 | +4.1 bp | +1.1 | 47% | 8/19 |
| low | ret30 | volatile half | 1 | 300 | _close | +10.9 bp | +9.3 bp | +3.6 | +2.9 bp | +1.0 | 50% | 7/19 |
| low | ret30 | volatile half | 1 | 300 | 240 | +10.9 bp | +9.3 bp | +3.6 | +2.9 bp | +1.0 | 50% | 7/19 |
| low | ret30 | volatile half | 1 | 300 | 90 | +10.9 bp | +9.3 bp | +3.6 | +2.9 bp | +1.0 | 50% | 7/19 |
| low | ret30 | all stocks | 1 | 300 | 90 | +10.7 bp | +9.1 bp | +3.5 | +2.7 bp | +0.9 | 50% | 7/19 |
| low | ret30 | all stocks | 1 | 300 | 240 | +10.7 bp | +9.1 bp | +3.5 | +2.7 bp | +0.9 | 50% | 7/19 |
| low | ret30 | all stocks | 1 | 300 | _close | +10.7 bp | +9.1 bp | +3.5 | +2.7 bp | +0.9 | 50% | 7/19 |
| low | ret5 | all stocks | 1 | 60 | _close | +12.6 bp | +10.5 bp | +2.2 | +4.6 bp | +0.9 | 50% | 11/19 |
| low | ret5 | volatile half | 1 | 60 | _close | +12.6 bp | +10.6 bp | +2.2 | +4.6 bp | +0.9 | 50% | 11/19 |
| low | prev_ret | most volatile tenth | 1 | 150 | 240 | +11.9 bp | +10.2 bp | +2.5 | +3.9 bp | +0.8 | 48% | 9/19 |
| low | prev_ret | most volatile tenth | 1 | 150 | _close | +11.9 bp | +10.2 bp | +2.5 | +3.9 bp | +0.8 | 48% | 9/19 |
| low | ret5 | most volatile tenth | 1 | 60 | _close | +11.7 bp | +9.8 bp | +2.1 | +3.7 bp | +0.7 | 50% | 11/19 |
| low | ret5 | all stocks | 1 | 30 | _close | +11.8 bp | +9.2 bp | +1.8 | +3.8 bp | +0.7 | 50% | 7/19 |

## Strongest stock-picking effect (excess over the average filtered stock), before costs

| Pick | Score | Filter | k | Minute | Hold | Gross | Excess over filter | Excess t | Net | Net t | Winning days | Years net > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | range_pos | all stocks | 3 | 330 | 10 | -1.9 bp | -1.9 bp | -9.0 | -15.9 bp | -49.1 | 11% | 0/19 |
| high | range_pos | volatile half | 3 | 330 | 10 | -2.2 bp | -2.2 bp | -8.2 | -16.2 bp | -43.0 | 18% | 0/19 |
| high | range_pos | all stocks | 1 | 330 | 10 | -2.4 bp | -2.5 bp | -7.5 | -10.4 bp | -25.3 | 25% | 0/19 |
| high | ret5 | all stocks | 3 | 30 | 10 | -5.5 bp | -5.5 bp | -7.3 | -19.5 bp | -22.6 | 31% | 0/19 |
| high | ret5 | volatile half | 3 | 30 | 10 | -5.5 bp | -5.5 bp | -7.3 | -19.5 bp | -22.7 | 31% | 0/19 |
| low | range_pos | all stocks | 3 | 330 | 10 | +1.6 bp | +1.6 bp | +7.1 | -12.4 bp | -39.4 | 18% | 0/19 |
| low | range_pos | volatile half | 3 | 330 | 10 | +1.8 bp | +1.8 bp | +7.1 | -12.2 bp | -33.8 | 24% | 0/19 |
| high | ret5 | most volatile tenth | 3 | 30 | 10 | -5.5 bp | -5.0 bp | -6.8 | -19.5 bp | -22.4 | 32% | 0/19 |
| low | range_pos | most volatile tenth | 3 | 30 | 10 | +2.5 bp | +2.9 bp | +6.7 | -11.5 bp | -20.4 | 34% | 0/19 |
| high | range_pos | volatile half | 3 | 270 | 10 | -2.4 bp | -1.7 bp | -6.5 | -16.4 bp | -45.2 | 19% | 0/19 |
| high | range_pos | all stocks | 3 | 330 | 240 | -2.8 bp | -3.2 bp | -6.5 | -16.8 bp | -19.9 | 29% | 0/19 |
| high | range_pos | all stocks | 3 | 330 | _close | -2.8 bp | -3.2 bp | -6.5 | -16.8 bp | -19.9 | 29% | 0/19 |
| high | range_pos | all stocks | 3 | 330 | 90 | -2.8 bp | -3.2 bp | -6.5 | -16.8 bp | -19.9 | 29% | 0/19 |
| high | range_pos | volatile half | 1 | 330 | 10 | -2.7 bp | -2.7 bp | -6.3 | -10.7 bp | -21.3 | 32% | 0/19 |
| high | ret5 | volatile half | 3 | 30 | 20 | -5.8 bp | -5.9 bp | -5.9 | -19.8 bp | -17.0 | 35% | 0/19 |
| high | ret5 | all stocks | 3 | 30 | 20 | -5.8 bp | -5.9 bp | -5.9 | -19.8 bp | -17.0 | 35% | 0/19 |
| low | ret5 | most volatile tenth | 3 | 30 | 10 | +4.1 bp | +4.5 bp | +5.8 | -9.9 bp | -10.7 | 39% | 1/19 |
| high | range_pos | most volatile tenth | 3 | 30 | 10 | -2.8 bp | -2.4 bp | -5.7 | -16.8 bp | -31.7 | 29% | 0/19 |
| low | ret5 | volatile half | 3 | 30 | 10 | +4.5 bp | +4.5 bp | +5.7 | -9.5 bp | -10.4 | 40% | 1/19 |
| low | ret5 | all stocks | 3 | 30 | 10 | +4.5 bp | +4.5 bp | +5.7 | -9.5 bp | -10.4 | 40% | 1/19 |

## Best net result per filter and number of picks

| Pick | Score | Filter | k | Minute | Hold | Gross | Excess over filter | Excess t | Net | Net t | Winning days | Years net > 0 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| low | ret5 | all stocks | 1 | 60 | _close | +12.6 bp | +10.5 bp | +2.2 | +4.6 bp | +0.9 | 50% | 11/19 |
| low | ret30 | all stocks | 3 | 150 | 240 | +9.5 bp | +7.6 bp | +4.0 | -4.5 bp | -1.8 | 47% | 5/19 |
| low | prev_ret | most volatile tenth | 1 | 240 | _close | +13.6 bp | +11.7 bp | +3.2 | +5.6 bp | +1.2 | 48% | 10/19 |
| low | prev_ret | most volatile tenth | 3 | 300 | 90 | +9.8 bp | +5.9 bp | +3.5 | -4.2 bp | -1.5 | 44% | 4/19 |
| low | ret5 | volatile half | 1 | 60 | _close | +12.6 bp | +10.6 bp | +2.2 | +4.6 bp | +0.9 | 50% | 11/19 |
| low | ret5 | volatile half | 3 | 30 | _close | +9.5 bp | +6.9 bp | +2.5 | -4.5 bp | -1.3 | 47% | 5/19 |
