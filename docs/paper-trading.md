# Paper trading (IBKR paper account)

Started 2026-10-22 at sergio's request. Code: `algotrader/live.py` (today's target), `algotrader/paper.py`
(records, decides, places orders), `config/paper.json` (no secrets). Why this configuration and how it
ranks against everything else tested: [`results/ranking.md`](../results/ranking.md).

## What it trades

Two **books** in the one paper account (IBKR gives one paper account per login), each with its own shares,
cash and record; their orders are netted per stock (details in STRATEGY.md, "Paper trading"):

- **Momentum, top 5, weekly, $15,000** (added 2026-10-28, sergio's candidate for real money): each Friday (the
  week's last NYSE session) after the close, rank today's members by their return from 12 months to 1 month ago
  (adjusted IBKR daily closes) and hold the top 5 at equal weight, trading only when the names change. Backtest
  2012 to Oct 2026: +26.9% a year (about +2.0% a month), Sharpe 0.97 [0.60, 1.35], volatility 29%, max drawdown
  -35.1%, worst month -27.8%, turnover about 820% a year; Fama-French alpha +9.6% (t 1.91, not significant). It is
  the market with about twice the swings, best of 1,226 tries; its lead over operating profitability is all after
  2018 (2012-18 +16.3% against +18.5% a year).
- **Operating profitability, top 10, monthly, $15,000** (about $1,500 a stock). Backtest 2012-2026 on the
  point-in-time S&P 500: CAGR +16.7% (about +1.29% a month), Sharpe 0.99 [0.55, 1.50], max drawdown -41.0%
  (Jan-Mar 2020, worst month -17.9%); SPY +15.0% (+1.17% a month), 0.93, -33.7% (rerun 2026-10-28 with the
  fundamentals fix; before it +16.0%, 0.98, -34.9%). Since 2019 +15.1% a year against SPY +17.2%. Fama-French alpha not significant: **plan on about the market**
  (6-8% a year is a sober expectation). The test is of execution and costs, not of an edge.
- Rule, as in the backtest: after the close of the last NYSE session of each month, rank today's S&P 500
  members by latest annual operating income / stockholders' equity (positive equity only; a report counts
  from the day after it was filed), take the top 10. If the names differ from those held, trade every
  holding to equal weight with **market-on-open** orders (filled at the next open, like the backtest);
  if they are the same, trade nothing.
- Differences from the backtest: today's member list (Wikipedia) instead of the archived lists; whole shares
  of a $1,500 slot (a stock over $1,500 a share is not bought and its slot stays in cash); 1% kept in cash;
  trims under $100 skipped; real IBKR fills and commissions.
- Today's target (2026-10-27 data, with the fundamentals fix): CL, GDDY, LYV, VRSK, IT, TPR, MA, APP, STX, AAPL
  (QCOM was there only through a stale 2019 equity value).

## Steps only sergio can do

1. **Open an IBKR Pro account** (Interactive Brokers Ireland) and fund it; the paper account exists only
   alongside a live one. Ask for US stock permissions. Choose **Tiered** commissions: on a $1,500 order the
   Fixed plan's $1 minimum is about 6.7 bp, Tiered about 3 bp.
2. **Create the paper account**: Client Portal > Settings > Paper Trading Account. Note its username (a
   separate login) and account id (starts with `DU`). **Reset its balance to $15,000** (or the euro
   equivalent; the runner refuses above `max_equity`). 2026-10-28: sergio reset it to about $30,000 for the two
   books (effective 2026-10-29); `max_equity` $45,000. Base currency USD if offered.
3. Optional: enable "share market data with the paper account". Not required: the runner sizes with the
   previous close, which delayed data gives for free.
4. Give the paper username and password to the machine that runs IB Gateway (step 5), in
   `~/.config/ibkr/ibc.ini`, chmod 600, **never in the repo or in a chat**. Approve the IBKR Mobile 2FA push
   on the first login and about once a week after.

## Gateway and cron (Claude does this once the account exists)

5. **IB Gateway + IBC on the Hermes VPS: installed 2026-10-26**, all in user space (the `hermes` user has
   no sudo; nothing installed system-wide, no port opened):
   - Java: Azul Zulu FX JDK **25.0.2** aarch64 in `~/ibkr/jdk` (Gateway 10.50's installer accepts exactly
     25.0.2; its bundled JRE is x64). Gateway 10.50 (stable) in `~/Jts/ibgateway/stable` (symlink `1050`),
     installed with `INSTALL4J_JAVA_HOME_OVERRIDE=~/ibkr/jdk INSTALL4J_DISABLE_BUNDLED_JRE=true sh
     ibgateway-stable-standalone-linux-x64.sh -q -dir ~/Jts/ibgateway/stable`.
   - Xvfb: Ubuntu `xvfb` and its five missing packages unpacked with `apt-get download` + `dpkg -x` into
     `~/ibkr/x`. Xvfb looks for `xkbcomp` in `/usr/bin`, so `~/ibkr/x/usr/bin/Xvfb-ibkr` is a copy with
     that path patched to `/tmp/xb` (a symlink to `~/ibkr/x/usr/bin`, recreated by the start script).
   - IBC 3.24.2 in `~/ibc` (`gatewaystart.sh`: version 1050, paper, Java above, 2FA timeout restarts);
     config in `~/.config/ibkr/ibc.ini` (chmod 600): paper, API port 4002, Read-Only API off, trusted IP
     127.0.0.1, other incoming connections rejected, daily restart 06:00 UTC (08:00 Madrid in winter).
     **`IbLoginId` and `IbPassword` are blank: sergio fills them in on the VPS.**
   - Start: `~/ibkr/start-gateway.sh` (starts Xvfb on :99 if needed, then IBC; refuses while the login is
     blank). Logs: `~/ibkr/gateway.log`, `~/ibc/logs/`. Tested: the Gateway opens its login dialog on ARM
     and IBC sets paper mode (about 300 MB RSS). Not yet started on reboot (needs a `@reboot` crontab entry).
   - 2026-10-27: sergio filled in the login; the Gateway logged in to DUR288542 (Simulated Trading, base EUR,
     €13,108). IBKR reports no USD exchange rate until the account holds dollars (and the paper account
     has no forex data), so the runner converts a euro account with the ECB daily reference rate until
     then, and refuses any other currency without a rate. Status and dry run work (account $14,666).
   - 2026-10-27 14:22 UTC (sergio's OK): EUR 12,950 converted to $14,435.11 (EUR.USD odd lot at 1.11468, commission
     EUR 1.79), so the stock purchases are paid in dollars instead of borrowing them (a euro account buying US stocks
     gets a dollar loan, about 4% a year). EUR 48.21 left; the trade settles 2026-10-29.
     With dollars held, IBKR reports the rate only as `$LEDGER-ExchangeRate` (EUR per USD, 0.8970318 = 1.11479
     USD per EUR), not as `ExchangeRate`; the runner now reads it before falling back to the ECB rate (which
     overstated the account by 0.4%: $14,662 instead of $14,612.50).
     The Gateway's API socket listens on all interfaces (it has no bind option); only 127.0.0.1 is trusted
     and other connections are rejected.
   - Stop: kill the PIDs from `pgrep -u hermes -f "[i]bcalpha.ibc.IbcGateway|[i]bcstart.sh|[g]atewaystart.sh"`
     (the brackets keep the pattern from matching the ssh shell itself).
6. **Dry run**: `uv run python -m algotrader.paper status`, then `uv run python -m algotrader.paper daily`
   (prints the orders it would place, places none).
7. **Cron** (VPS clock is UTC; 21:30 UTC is after the US close in summer and winter):
   ```
   30 21 * * 1-5 cd ~/algotrader-work && ~/.local/bin/uv run python -m algotrader.paper daily --submit >> data/paper/cron.log 2>&1
   ```
   The first run builds both books at the next open; after that operating profitability decides at month ends and
   momentum at week ends. The Friday run of the momentum book downloads about 500 daily histories from IBKR
   (kept in `data/paper/prices/`).
8. **Telegram** (done 2026-10-27, sergio's OK): the `notify-telegram` sender and its bot config are on the VPS in
   `~/.config/telegram-notify/` (directory 700, config 600, never in the repo); `notify_cmd` in `config/paper.json` points
   to its `send.sh`. Every run's message (daily summary, month-end plan and orders) and every error, including a Gateway
   that cannot be reached, is sent. Not covered: a cron job that fails before Python starts (check `cron.log`).

## Safety

The runner refuses to trade if the account id does not start with `DU` in paper mode (and the reverse in
live mode), the account is worth more than `max_equity` or has less unallocated cash than a new book's
capital, orders are still open, the books' shares do not add up to the account's positions, more than
`max_orders` orders or a buy over 1.5 of a book's slots would go out, or a target stock has no price; each book
decides at most once a day;
`touch data/paper/STOP` stops it. Without `--submit` it never places an order, and it connects read-only.

## What gets recorded (`data/paper/`, on the machine that runs it)

`equity.csv` (account value in USD, cash, positions, the base currency, the value in it, the USD rate used and
its source, `ibkr` or `ecb`, every weekday), `books.csv` (each book's value, cash and shares, every weekday),
`fills.csv` (IBKR fills with their commissions and order reference), `orders.csv` (net orders placed, with the
close used for sizing), `target_<signal>_<day>.json` (each decision's top N and scores), `prices/` (momentum's
closes), `state.json` (each book's shares, cash, target, last decision; orders waiting for their fills). Compare monthly with SPY and with the backtest's costs (2 bp
half-spread + commission): paper fills are top of book only, so they flatter live trading.
