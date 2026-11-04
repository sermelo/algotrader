"""scripts/start_gateway.sh with stub process tools: starts, leaves a healthy Gateway alone, restarts one
that runs without its API port, waits for a fresh one, and reports a start that never opens the port."""
import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "start_gateway.sh"

STUBS = {
    # gateway pattern: running if the flag exists; Xvfb: running if its flag exists
    "pgrep": '[ -n "$2" ] || exit 2; case "$*" in *Xvfb*) [ -e "$FAKE/xvfb" ];; *) [ -e "$FAKE/running" ] && echo 123;; esac',
    "pkill": 'echo "pkill $*" >> "$FAKE/log"; rm -f "$FAKE/running" "$FAKE/listening"',
    "ss": '[ -e "$FAKE/listening" ] && echo "LISTEN 0 50 *:4002 *:*"; true',
    "ps": 'cat "$FAKE/etimes"',
    "setsid": 'echo "setsid $*" >> "$FAKE/log"; case "$*" in *gatewaystart*) '
              '[ -e "$FAKE/opens_port" ] && touch "$FAKE/running" "$FAKE/listening";; esac; true',
    "sleep": "/bin/sleep 0.05",                     # the started Gateway runs in the background
}


def run(tmp_path, *flags, etimes=0, login=True, user="algotrader"):
    home, fake, bin_ = tmp_path / "home", tmp_path / "fake", tmp_path / "bin"
    for d in (home / ".config" / "ibkr", home / "ibkr", fake, bin_):
        d.mkdir(parents=True, exist_ok=True)
    (home / ".config" / "ibkr" / "ibc.ini").write_text("IbLoginId=someone\n" if login else "IbLoginId=\n")
    for name, body in STUBS.items():
        (bin_ / name).write_text("#!/bin/sh\n" + body + "\n")
        (bin_ / name).chmod(0o755)
    notify = bin_ / "notify"
    notify.write_text('#!/bin/sh\necho "notify $1" >> "$FAKE/log"\n')
    notify.chmod(0o755)
    for f in flags:
        (fake / f).touch()
    (fake / "etimes").write_text(f"{etimes}\n")
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "HOME": str(home), "FAKE": str(fake),
           "IBKR_XB": str(tmp_path / "xb"), "IBKR_NOTIFY": str(notify)}
    if user:
        env["USER"] = user
    p = subprocess.run(["sh", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30)
    log = (fake / "log").read_text() if (fake / "log").exists() else ""
    return p.returncode, p.stdout, log


def test_starts_a_missing_gateway_and_waits_for_the_port(tmp_path):
    code, out, log = run(tmp_path, "opens_port")
    assert code == 0
    assert "gateway started" in out and "port 4002 listening after" in out
    assert "setsid " in log and "gatewaystart.sh" in log and "pkill" not in log
    assert (tmp_path / "xb").is_symlink()


def test_leaves_a_listening_gateway_alone(tmp_path):
    code, out, log = run(tmp_path, "running", "listening", "xvfb", etimes=50_000)
    assert code == 0 and "already running" in out
    assert log == ""


def test_restarts_a_gateway_running_without_its_port(tmp_path):
    code, out, log = run(tmp_path, "running", "xvfb", "opens_port", etimes=3_600)
    assert code == 0
    assert "not listening: restarting" in out and "listening after" in out
    assert log.index("pkill") < log.index("gatewaystart.sh")


def test_without_user_set_as_under_cron(tmp_path):
    code, out, log = run(tmp_path, "running", "listening", "xvfb", etimes=50_000, user=None)
    assert code == 0 and "already running" in out
    assert log == ""


def test_a_gateway_still_logging_in_is_not_killed(tmp_path):
    code, out, log = run(tmp_path, "running", "xvfb", etimes=40)
    assert code == 0 and "left alone" in out
    assert log == ""


def test_a_start_that_never_opens_the_port_sends_a_message(tmp_path):
    code, out, log = run(tmp_path, "xvfb")
    assert code == 2
    assert "notify IB Gateway (VPS): port 4002 still closed" in log


def test_refuses_without_a_login(tmp_path):
    code, out, log = run(tmp_path, login=False)
    assert code == 1 and "no IbLoginId" in out
    assert log == ""
