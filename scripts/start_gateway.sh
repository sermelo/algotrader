#!/bin/sh
# Start Xvfb (user-space copy, display :99) and IB Gateway via IBC (paper, API on 127.0.0.1:4002).
# Deployed on the VPS as ~/ibkr/start-gateway.sh (cron: @reboot and 21:15 UTC on weekdays).
# A Gateway process that runs but has not opened the API port for GRACE seconds (e.g. stuck logged
# out after IBKR refused a login, as on 2026-10-31) is killed and started again. If the port is
# still closed WAIT seconds after a start, a Telegram message is sent.
PORT=${IBKR_PORT:-4002}
GRACE=${IBKR_GRACE:-300}
WAIT=${IBKR_WAIT:-180}
XB=${IBKR_XB:-/tmp/xb}
NOTIFY=${IBKR_NOTIFY:-$HOME/.config/telegram-notify/send.sh}
PATTERN="[i]bcalpha.ibc.IbcGateway|[i]bcstart.sh|[g]atewaystart.sh"

stamp() { echo "$(date -u '+%F %T') $*"; }
listening() { ss -ltnH "sport = :$PORT" | grep -q .; }
gateway_pid() { pgrep -u "$USER" -f "ibcalpha.ibc.IbcGateway" | head -n 1; }
notify() { [ -x "$NOTIFY" ] && "$NOTIFY" "$*" >/dev/null 2>&1; stamp "$*"; }

grep -q "^IbLoginId=." "$HOME/.config/ibkr/ibc.ini" || { stamp "no IbLoginId in ~/.config/ibkr/ibc.ini"; exit 1; }

pid=$(gateway_pid)
if [ -n "$pid" ]; then
  if listening; then stamp "gateway already running"; exit 0; fi
  age=$(ps -o etimes= -p "$pid" | tr -d ' ')
  if [ "${age:-0}" -lt "$GRACE" ]; then stamp "gateway started ${age}s ago, port $PORT not open yet: left alone"; exit 0; fi
  stamp "gateway running ${age}s but port $PORT not listening: restarting"
  pkill -u "$USER" -f "$PATTERN"
  sleep 10
  pgrep -u "$USER" -f "$PATTERN" >/dev/null && pkill -9 -u "$USER" -f "$PATTERN"
fi

X=$HOME/ibkr/x
export LD_LIBRARY_PATH=$X/usr/lib/aarch64-linux-gnu
# Patched Xvfb looks for xkbcomp in /tmp/xb (no root to install it in /usr/bin).
[ "$(readlink "$XB")" = "$X/usr/bin" ] || { rm -f "$XB"; ln -s "$X/usr/bin" "$XB"; }
if ! pgrep -u "$USER" -x Xvfb-ibkr >/dev/null; then
  setsid "$X/usr/bin/Xvfb-ibkr" :99 -screen 0 1024x768x24 -nolisten tcp >"$HOME/ibkr/xvfb.log" 2>&1 &
  sleep 2
fi
unset LD_LIBRARY_PATH
export DISPLAY=:99
setsid nohup "$HOME/ibc/gatewaystart.sh" -inline >>"$HOME/ibkr/ibc-console.log" 2>&1 &
stamp "gateway started"

waited=0
while [ "$waited" -lt "$WAIT" ]; do
  sleep 10
  waited=$((waited + 10))
  if listening; then stamp "port $PORT listening after ${waited}s"; exit 0; fi
done
notify "IB Gateway (VPS): port $PORT still closed ${WAIT}s after a start; check ~/ibc/logs (login refused?)"
exit 2
