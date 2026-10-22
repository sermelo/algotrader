# Paper trading (IBKR paper account)

Started 2026-10-22 at sergio's request. Code: `algotrader/live.py` (today's target), `algotrader/paper.py`
(records, decides, places orders), `config/paper.json` (no secrets). Why this configuration and how it
ranks against everything else tested: [`results/ranking.md`](../results/ranking.md).

## What it trades

- **Operating profitability, top 10, monthly, $15,000** (about $1,500 a stock). Backtest 2012-2026 on the
  point-in-time S&P 500: CAGR +16.0% (about +1.25% a month), Sharpe 0.98 [0.56, 1.44], max drawdown -34.9%;
  SPY +15.0% (+1.17% a month), 0.93, -33.7%. Fama-French alpha not significant: **plan on about the market**
  (6-8% a year is a sober expectation). The test is of execution and costs, not of an edge.
- Rule, as in the backtest: after the close of the last NYSE session of each month, rank today's S&P 500
  members by latest annual operating income / stockholders' equity (positive equity only; a report counts
  from the day after it was filed), take the top 10. If the names differ from those held, trade every
  holding to equal weight with **market-on-open** orders (filled at the next open, like the backtest);
  if they are the same, trade nothing.
- Differences from the backtest: today's member list (Wikipedia) instead of the archived lists; whole shares
  of a $1,500 slot (a stock over $1,500 a share is not bought and its slot stays in cash); 1% kept in cash;
  trims under $100 skipped; real IBKR fills and commissions.
- Today's target (2026-10-21 data): CL, GDDY, LYV, VRSK, IT, TPR, QCOM, MA, APP, STX.

## Steps only sergio can do

1. **Open an IBKR Pro account** (Interactive Brokers Ireland) and fund it; the paper account exists only
   alongside a live one. Ask for US stock permissions. Choose **Tiered** commissions: on a $1,500 order the
   Fixed plan's $1 minimum is about 6.7 bp, Tiered about 3 bp.
2. **Create the paper account**: Client Portal > Settings > Paper Trading Account. Note its username (a
   separate login) and account id (starts with `DU`). **Reset its balance to $15,000** (or the euro
   equivalent; the runner refuses above `max_equity`, $30,000). Base currency USD if offered.
3. Optional: enable "share market data with the paper account". Not required: the runner sizes with the
   previous close, which delayed data gives for free.
4. Give the paper username and password to the machine that runs IB Gateway (step 5), in
   `~/.config/ibkr/ibc.ini`, chmod 600, **never in the repo or in a chat**. Approve the IBKR Mobile 2FA push
   on the first login and about once a week after.

## Gateway and cron (Claude does this once the account exists)

5. **IB Gateway + IBC** (github.com/IbcAlpha/IBC) on the Hermes VPS, listening on localhost only (port
   4002, no port opened outward). The VPS is ARM (aarch64) with no Java: it needs an ARM Java 17 with JavaFX
   (e.g. Azul Zulu FX), Xvfb for the virtual display, and the Gateway jars run with that Java. Check with the
   `Hermes` session first (packages to install, memory: Gateway needs about 1 GB). Alternative: run it on
   this machine (x64, standard installer) if it stays on; it has little free memory.
   API settings: enable socket clients, **uncheck Read-Only API**, trusted IP 127.0.0.1, auto-restart outside
   15:30-22:00 Madrid.
6. **Dry run**: `uv run python -m algotrader.paper status`, then `uv run python -m algotrader.paper daily`
   (prints the orders it would place, places none).
7. **Cron** (VPS clock is UTC; 21:30 UTC is after the US close in summer and winter):
   ```
   30 21 * * 1-5 cd ~/algotrader-work && ~/.local/bin/uv run python -m algotrader.paper daily --submit >> data/paper/cron.log 2>&1
   ```
   The first run builds the portfolio at the next open; after that it decides only at month ends.
8. **Telegram**: set `notify_cmd` in `config/paper.json` to a sender on that machine (the
   `notify-telegram` skill lives on sergio's machine only; copying its bot token to the VPS needs his OK).

## Safety

The runner refuses to trade if the account id does not start with `DU` in paper mode (and the reverse in
live mode), the account is worth more than `max_equity`, orders are still open, more than `max_orders`
orders or a buy over 1.5 slots would go out, or a target stock has no price; it decides at most once a day;
`touch data/paper/STOP` stops it. Without `--submit` it never places an order, and it connects read-only.

## What gets recorded (`data/paper/`, on the machine that runs it)

`equity.csv` (account value in USD, cash, positions, every weekday), `fills.csv` (IBKR fills with their
commissions), `orders.csv` (orders placed, with the close used for sizing), `target_<day>.json` (each
decision's top 10 and scores), `state.json`. Compare monthly with SPY and with the backtest's costs (2 bp
half-spread + commission): paper fills are top of book only, so they flatter live trading.
