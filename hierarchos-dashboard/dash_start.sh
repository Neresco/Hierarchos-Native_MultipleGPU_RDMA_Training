#!/usr/bin/env bash
#
# dash_start.sh — start/stop/restart/status the Hierarchos dashboard (WebUI).
#
# Usage:
#   ./dash_start.sh              # start (default)
#   ./dash_start.sh start        # start
#   ./dash_start.sh stop         # stop
#   ./dash_start.sh restart      # restart
#   ./dash_start.sh status       # show listening port + pid
#   ./dash_start.sh nodes        # show registered nodes (ranks/GPUs/RDMA)
#
# Environment:
#   DASH_PORT  port to serve on (default 7788)
#   DASH_HOST  bind address (default 0.0.0.0)
#
set -u

# Resolve this script's directory so it works from any location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVER="$SCRIPT_DIR/server.py"
LOG="$SCRIPT_DIR/dash.log"
NODES="$SCRIPT_DIR/nodes.json"
PORT="${DASH_PORT:-7788}"

cmd="${1:-start}"

is_running() {
  pgrep -f "hierarchos-dashboard/server.py" >/dev/null 2>&1
}

do_start() {
  if is_running; then
    echo "dashboard already running (see status)"
    return 0
  fi
  if [[ ! -f "$SERVER" ]]; then
    echo "error: server.py not found at $SERVER" >&2
    return 1
  fi
  # Ensure a nodes file exists (server seeds DEFAULT_NODES if missing).
  [[ -f "$NODES" ]] || echo "note: $NODES not found; server will use built-in defaults"
  mkdir -p "$SCRIPT_DIR"
  setsid nohup python3 "$SERVER" > "$LOG" 2>&1 < /dev/null &
  disown 2>/dev/null || true
  sleep 2
  if is_running; then
    echo "dashboard started -> http://0.0.0.0:${PORT}/  (pid $(pgrep -f 'hierarchos-dashboard/server.py' | head -1))"
  else
    echo "dashboard failed to start; check $LOG" >&2
    tail -5 "$LOG" 2>/dev/null
    return 1
  fi
}

do_stop() {
  if is_running; then
    pkill -f "hierarchos-dashboard/server.py" 2>/dev/null
    sleep 1
    is_running && echo "still running; force-killing" && pkill -9 -f "hierarchos-dashboard/server.py"
    echo "dashboard stopped"
  else
    echo "dashboard not running"
  fi
}

do_status() {
  if is_running; then
    local pid; pid="$(pgrep -f 'hierarchos-dashboard/server.py' | head -1)"
    echo "running: pid $pid -> http://0.0.0.0:${PORT}/"
    (ss -tlnp 2>/dev/null | grep ":${PORT} ") || true
  else
    echo "not running"
  fi
}

do_nodes() {
  if [[ -f "$NODES" ]]; then
    python3 - "$NODES" <<'PY'
import json, sys
nodes = json.load(open(sys.argv[1]))
print(f"{'key':<12} {'name':<22} {'ip':<16} {'rank':<5} {'dev':<4} {'rdma':<12} remote")
for n in nodes:
    print(f"{n.get('key',''):<12} {n.get('name',''):<22} {n.get('ip',''):<16} "
          f"{str(n.get('rank','-')):<5} {str(n.get('dev_index','-')):<4} "
          f"{n.get('rdma_device',''):<12} {n.get('remote') or '(local)'}")
PY
  else
    echo "no $NODES (server seeds defaults on first start)"
  fi
}

# --- node editing helpers (operate on nodes.json with python) ---

node_add() {
  read -rp "Node key (e.g. pc3-gpu0): " key; key="${key,,}"
  read -rp "Display name: " name; name="${name:-$key}"
  read -rp "IP: " ip
  read -rp "SSH remote (blank = local, e.g. user@ip): " remote
  read -rp "GPU device index: " dev
  read -rp "RDMA device (e.g. rocep198s0): " rdma
  read -rp "Ring rank (0-based): " rank
  [[ -z "$ip" || -z "$rdma" || -z "$rank" ]] && { echo "IP, RDMA and rank are required"; return 1; }
  python3 - "$NODES" <<PY
import json, sys
p = sys.argv[1]
try: nodes = json.load(open(p))
except Exception: nodes = []
remote = "$remote" if "$remote" else None
nodes.append({"key": "$key", "name": "$name", "ip": "$ip", "remote": remote,
              "dev_index": $dev, "rdma_device": "$rdma", "rank": $rank})
json.dump(nodes, open(p, "w"), indent=2)
print(f"added {len(nodes)}th node -> $NODES")
PY
}

node_remove() {
  do_nodes
  read -rp "Node key to remove: " key
  python3 - "$NODES" <<PY
import json, sys
p = sys.argv[1]
nodes = json.load(open(p))
before = len(nodes)
nodes = [n for n in nodes if n.get("key") != "$key"]
json.dump(nodes, open(p, "w"), indent=2)
print(f"removed: {before - len(nodes)} node(s)")
PY
}

node_test_ssh() {
  do_nodes
  read -rp "Node key to test SSH to: " key
  remote="$(python3 -c "import json,sys; ns=json.load(open('$NODES')); print(next((n.get('remote') for n in ns if n.get('key')=='$key'), ''))")"
  if [[ -z "$remote" ]]; then echo "node '$key' is local (no SSH test needed)"; return; fi
  if ssh -o ConnectTimeout=8 -o BatchMode=yes "$remote" 'hostname; echo ssh-ok'; then
    echo "SSH to $remote OK"
  else
    echo "SSH to $remote FAILED (set up key auth: ssh-copy-id $remote)"
  fi
}

menu() {
  while true; do
    echo
    echo "================ Hierarchos Dashboard ================"
    echo "  DASHBOARD: ${PORT}/  ($([[ -f $NODES ]] && echo "$(python3 -c "import json;print(len(json.load(open('$NODES'))))" 2>/dev/null) nodes") registered)"
    echo "-------------------------------------------------------"
    echo " 1) Start dashboard"
    echo " 2) Stop dashboard"
    echo " 3) Restart dashboard"
    echo " 4) Dashboard status"
    echo " 5) List nodes (GPU / IP / RDMA / rank)"
    echo " 6) Add a node (GPU endpoint)"
    echo " 7) Remove a node"
    echo " 8) Test SSH to a node"
    echo " 9) Open dashboard URL (xdg-open)"
    echo " 0) Exit"
    echo "======================================================="
    read -rp "Choose [0-9]: " choice
    case "$choice" in
      1) do_start ;;
      2) do_stop ;;
      3) do_stop; do_start ;;
      4) do_status ;;
      5) do_nodes ;;
      6) node_add ;;
      7) node_remove ;;
      8) node_test_ssh ;;
      9) xdg-open "http://0.0.0.0:${PORT}/" 2>/dev/null || echo "open http://0.0.0.0:${PORT}/ in a browser" ;;
      0) echo "bye"; break ;;
      *) echo "invalid choice" ;;
    esac
  done
}

case "$cmd" in
  start)   do_start ;;
  stop)    do_stop ;;
  restart) do_stop; do_start ;;
  status)  do_status ;;
  nodes)   do_nodes ;;
  add)     node_add ;;
  remove)  node_remove ;;
  menu)    menu ;;
  *) menu ;;  # default: interactive menu
esac
