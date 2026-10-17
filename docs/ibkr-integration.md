# IBKR integration notes (reference)

Saved 2026-10-17 at sergio's request, from the investment-portfolio Claude session.
Gathered from IBKR docs and campus pages plus community sources; items marked
(unverified) still need checking against IBKR's own pages or the account. Nothing here
is implemented yet; the backtester's cost model and live-trading constraints are in
`ROADMAP.md` ("Constraints for trading it live").

## 1. Which interface to use

- Use the **TWS API via IB Gateway** with Python `ib_async` for the bot: fullest order
  types, the standard for automated trading.
- The alternative, the Client Portal Web API (REST), needs its own Java gateway with a
  **daily manual browser login** for retail. IBKR says automating that login is not
  supported (unofficial tools such as IBeam do it anyway). OAuth is institutional-only;
  OAuth 2.0 for individuals was "being considered" as of 2025-08. Prefer the TWS API.
- **MCP is not for the bot.** IBKR's official claude.ai connector never sends orders: it
  creates an "AI Instruction" in IBKR Desktop (Portfolio > AI Instructions) that a human
  must Review & Submit. Fine for read-only questions or manual drafts. Community MCP
  servers wrap TWS/Gateway and default to read-only or confirmation; not needed.

## 2. Account and authentication

- Entity: Interactive Brokers Ireland (CNMV-registered EU firm no. 5023). Plan: IBKR Pro
  (Lite is US residents only). Needs US stock trading permissions; margin vs cash
  account matters for PDT (section 6).
- 2FA via IBKR Mobile (IB Key) is effectively mandatory.
- IB Gateway: the daily auto-restart usually does not need 2FA, but there is a **weekly
  re-authentication** (cycle starts Monday; in practice a 2FA prompt on Sunday or
  Monday/Tuesday), so sergio approves one push a week. "Login failed = Soft token=0
  received" means a manual login is needed.
- Unattended running: **IBC** (github.com/IbcAlpha/IBC) automates the Gateway login and
  restarts. config.ini options: `ExitAfterSecondFactorAuthenticationTimeout`,
  `SecondFactorAuthenticationExitInterval` (retry when the 2FA push is missed). Docker
  images bundling Gateway + IBC exist (e.g. gnzsnz/ib-gateway-docker; unverified).
- Gateway settings "Never lock" + "Auto restart"; schedule the restart outside US market
  hours (US session 15:30-22:00 Madrid time). IBKR also has a nightly server reset with
  brief disconnects (exact window unverified): the bot must handle reconnects.
- Alert sergio on disconnect or pending 2FA (`~/.claude/skills/notify-telegram/send.sh`).

## 3. Paper trading first

- Every live account gets a free paper login (account ids start with "DU"), with
  separate credentials (Client Portal > Settings > Paper Trading Account). A live
  account must be opened and funded first.
- Default ports: TWS live 7496 / paper 7497; **Gateway live 4001 / paper 4002**. Do not
  rely on the port: on connect read the account id and refuse to trade if it is not
  DU... in paper mode (and the reverse in live mode).
- Paper balance can be reset (e.g. to $1M) in Client Portal Account Settings: set it to
  what sergio would really allocate, so sizing and buying power match.
- Market data in paper: enable "share live market data with paper account". Shared
  data cannot be used by live and paper at the same time; the paper session loses data
  if the live user logs in elsewhere (phone, Client Portal); IBKR KB says both sessions
  must be on the same device. Workaround: a second live username (pays its own data fees).
- One username cannot be logged into two trading apps at once (TWS/mobile login can kick
  the Gateway).
- Paper commissions: believed to use the live plan (Tiered/Fixed); verify with a test
  order. Fills are optimistic (top of book only, no queue or depth; stops and complex
  orders simulated; VWAP/Auction/RFQ/Pegged-to-Market unsupported). Log expected price vs
  fill on every trade.

## 4. Market data (needed for live intraday)

- Without a subscription the API gives ~15-minute delayed data.
- Non-professional: "US Securities Snapshot and Futures Value Bundle" $10/month, waived in
  any month with >= $30 commissions (snapshots only); "US Equity and Options Add-On
  Streaming Bundle" ~$4.50/month, no waiver (streaming; source over a year old, verify).
  Expect ~$4.50-14.50 a month. Declare non-professional status.
- Default **100 market data lines** (unverified for this account): how many tickers can
  stream at once; more need commission volume or "Quote Booster" packs.

## 5. API limits and settings

- Message rate: max requests/s = market data lines / 2 -> **50 msg/s** by default;
  exceeding it 3 times terminates the session (or TWS paces, depending on "Reject
  messages above maximum allowed message rate").
- Historical data pacing (community-reported): no identical request within 15 s, no 6+
  requests for the same contract/exchange/tick type within 2 s, <= 60 requests per 10
  minutes. Keep using the existing data sources for backtests.
- Gateway/TWS API settings: enable "ActiveX and Socket Clients", **uncheck "Read-Only
  API"** to send orders, allow localhost only (trusted IP 127.0.0.1), enable "Bypass Order
  Precautions for API orders" (name unverified) so precaution dialogs do not block API
  orders. Unique clientId per connection; manage nextValidId.
- Commission per fill comes back in `commissionReport`: store it to compare real costs
  with the Fixed and Tiered models.

## 6. Constraints

See `ROADMAP.md`: PRIIPs (no US-domiciled ETFs for EU retail), PDT status at IBKR
unconfirmed (SEC abolished it from 2026-06-04, brokers have until 2027-10-20), Spanish
two-month rule, keep a USD balance.
For a roughly 1,000-euro account see `R&D/small-account-intraday.md`: likely a cash account (T+1
settlement, one round trip a day, free-riding rule), Tiered commissions, convert to dollars once.

## 7. Suggested safety controls for going live

Max daily loss -> stop trading; max position size and max orders per minute; flatten
before the close if the strategy is intraday-only; a kill switch; refuse to start if the
account mode does not match the config; Telegram alerts on errors, disconnects and fills.

## Sources

interactivebrokers.com/campus (paper trading lessons, "How to Set Up Interactive Brokers
and ORATS in Claude", Client Portal Gateway lessons); interactivebrokers.com/docs
(daily/weekly re-authentication, TWS API pacing, logging into multiple applications);
IBKR market data pricing page; github.com/IbcAlpha/IBC; pickmytrade.io help pages;
QuantConnect and elitetrader forums.
