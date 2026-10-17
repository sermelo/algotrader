# Replay of the best-of-day trades on the raw minute bars

1,119 of 1,119 trades in 53 sampled months (every 3rd month) could be replayed (a trade needs a print in its entry minute). Hold 90 minutes, cost 8 bp.

Check: the plain fill reproduces the dataset's label on 100.0% of trades (mean difference 0.00 bp).

| Fill | Gross per trade | Net of 8 bp | Net t | Median gross | Winning trades after cost |
|---|---|---|---|---|---|
| plain: buy the entry minute's open, sell the exit minute's open | +39.7 bp | +31.7 bp | +6.3 | +38.5 bp | 60% |
| entry 1 minute late | +36.6 bp | +28.6 bp | +5.8 | +36.1 bp | 59% |
| entry 2 minutes late | +36.5 bp | +28.5 bp | +5.7 | +36.4 bp | 58% |
| entry 5 minutes late | +35.9 bp | +27.9 bp | +5.7 | +33.2 bp | 59% |
| slippage: a quarter of each bar's range against us | +28.6 bp | +20.6 bp | +4.1 | +29.5 bp | 57% |
| worst case: buy the entry minute's high, sell the exit minute's low | +15.5 bp | +7.5 bp | +1.5 | +18.4 bp | 54% |
