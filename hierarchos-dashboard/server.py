#!/usr/bin/env python3
"""Hierarchos-Native training dashboard (dynamic multi-node aware).

Zero-dependency (Python stdlib) web server that shows live training state and
GPU/CPU hardware status for a configurable set of nodes.

Run:  python3 server.py   (listens on 0.0.0.0:7788)

Node registry lives in nodes.json next to this file. Each node:
  {"key": "primary", "name": "PC1 · Primary", "ip": "192.168.1.101",
   "remote": "lunarbuntu@192.168.1.101" (or null for local), "role": "primary"}
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import time
import urllib.parse

PORT = int(os.environ.get("DASH_PORT", "7788"))
HOME = os.path.expanduser("~")
WORK = os.path.join(HOME, "Hierarchos-Native-work")
CLI = os.path.join(HOME, "Hierarchos-Native/hierarchos-native-cli/target/release/hierarchos-native-cli")
LOG = os.path.join(HOME, "hierarchos-full.log")
TRAIN_LOG = os.path.join(HOME, "hierarchos-train-run.log")
NODES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nodes.json")

DEFAULT_NODES = [
    {"key": "primary", "name": "PC1 · Primary", "ip": "192.168.1.101",
     "remote": "lunarbuntu@192.168.1.101", "role": "primary"},
    {"key": "replica", "name": "PC2 · Replica", "ip": "192.168.100.20",
     "remote": None, "role": "replica"},
]

# per-node server-side epoch tracker for ETA
_epoch_tracker = {}


def _run(cmd, timeout=15):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:
        return f"error: {e}"


def _ssh_run(remote, cmd, timeout=20):
    if not remote:
        return _run(cmd, timeout)
    full_cmd = f'export PATH="$PATH:/opt/rocm/bin:/usr/rocm/bin:/opt/rocm/lib"; {cmd}'
    try:
        r = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", remote, full_cmd],
            capture_output=True, text=True, timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def load_nodes():
    try:
        with open(NODES_FILE) as f:
            nodes = json.load(f)
        if isinstance(nodes, list) and nodes:
            return nodes
    except Exception:
        pass
    save_nodes(DEFAULT_NODES)
    return list(DEFAULT_NODES)


def save_nodes(nodes):
    with open(NODES_FILE, "w") as f:
        json.dump(nodes, f, indent=2)


def node_run(node):
    remote = node.get("remote")
    if remote:
        return lambda cmd, t: _ssh_run(remote, cmd, t)
    return lambda cmd, t: _run(cmd, t)


def training_running(remote=None):
    cmd = 'pgrep -f "[h]ierarchos-vulkan-train" >/dev/null 2>&1 && echo 1 || echo 0'
    return _ssh_run(remote, cmd).strip() == "1"


def _read_log(remote, key=None):
    # Per-node log: ~/hierarchos-logs/<key>.log (each GPU/rank has its own).
    # Falls back to the shared TRAIN_LOG / LOG for legacy runs.
    if key:
        p = os.path.join(HOME, "hierarchos-logs", f"{key}.log")
        if remote:
            out = _ssh_run(remote, f"cat {p} 2>/dev/null")
            if out.strip():
                return out
        else:
            if os.path.exists(p):
                try:
                    with open(p, "r", errors="replace") as f:
                        return f.read()
                except Exception:
                    pass
    if remote:
        return _ssh_run(remote, f"cat {TRAIN_LOG} 2>/dev/null || cat {LOG} 2>/dev/null")
    for p in (TRAIN_LOG, LOG):
        if os.path.exists(p):
            try:
                with open(p, "r", errors="replace") as f:
                    return f.read()
            except Exception:
                continue
    return ""


def _parse_epochs(content, info):
    epoch_re = re.compile(
        r"epoch\s+(\d+)/(\d+):\s+batches=(\d+)\s+optimizer_step=(\d+)\s+live_lr=([\d.eE+-]+)\s+mean_recorded_loss=([\d.]+)"
    )
    for m in epoch_re.finditer(content):
        ep, tot, batches, step, lr, loss = m.groups()
        info["total_epochs"] = int(tot)
        info["epochs_done"] = max(info["epochs_done"], int(ep))
        info["optimizer_step"] = max(info["optimizer_step"], int(step))
        info["live_lr"] = float(lr)
        info["last_loss"] = float(loss)
        info["loss_history"].append({"epoch": int(ep), "loss": float(loss), "step": int(step)})
    for m in re.finditer(r"optimizer_step=(\d+)", content):
        info["optimizer_step"] = max(info["optimizer_step"], int(m.group(1)))


def parse_training():
    out = {}
    for node in load_nodes():
        key = node["key"]
        info = {
            "running": training_running(node.get("remote")),
            "epochs_done": 0, "total_epochs": 0, "optimizer_step": 0,
            "last_loss": None, "live_lr": None, "loss_history": [],
            "eta_seconds": None, "elapsed_seconds": None,
            "model_dir": os.path.join(WORK, "out-engram"),
            "model_saved": os.path.exists(os.path.join(WORK, "out-engram", "model.safetensors")),
        }
        _parse_epochs(_read_log(node.get("remote"), key), info)
        tracker = _epoch_tracker.setdefault(key, {})
        if info["epochs_done"] >= 1 and info["total_epochs"]:
            now = time.time()
            if info["epochs_done"] not in tracker:
                tracker[info["epochs_done"]] = now
            if len(tracker) > 1 and min(tracker.values()) < now - 7200:
                _epoch_tracker[key] = {k: v for k, v in tracker.items() if v > now - 7200}
                tracker = _epoch_tracker[key]
            completed = sorted(tracker.items())
            if len(completed) >= 2:
                first_ep, first_t = completed[0]
                last_ep, last_t = completed[-1]
                if last_ep > first_ep:
                    per_epoch = (last_t - first_t) / (last_ep - first_ep)
                    remaining = info["total_epochs"] - info["epochs_done"]
                    info["eta_seconds"] = int(per_epoch * remaining)
                    info["elapsed_seconds"] = int(last_t - first_t)
        info["current_epoch"] = (info["epochs_done"] + 1) if (info["running"] and info["epochs_done"] < info["total_epochs"]) else info["epochs_done"]
        out[key] = info
    return out


def _gtt_for_card(run, card):
    out = run(f"cat /sys/class/drm/{card}/device/mem_info_gtt_total /sys/class/drm/{card}/device/mem_info_gtt_used 2>/dev/null", 5)
    parts = out.split()
    if len(parts) >= 2:
        try:
            return round(int(parts[0]) / 1e9, 1), round(int(parts[1]) / 1e9, 2)
        except ValueError:
            return None, None
    return None, None


def _gpu_indices(run):
    out = run("rocm-smi --showproductname 2>/dev/null | grep 'Card Series'", 15)
    idxs, names = [], {}
    for line in out.splitlines():
        m = re.match(r"GPU\[(\d+)\]\s*:\s*Card Series:\s*(.+)", line)
        if m:
            idxs.append(int(m.group(1)))
            names[int(m.group(1))] = m.group(2).strip()
    return idxs, names


def _collect_gpus(run, idxs, names, gtt):
    out = run("rocm-smi --showuse --showmeminfo vram --showtemp --showpower 2>/dev/null", 20)
    gpus = []
    for idx in idxs:
        gtt_total, gtt_used = (_gtt_for_card(run, f"card{idx}") if gtt else (None, None))
        g = {"index": idx, "name": names.get(idx, f"GPU{idx}"), "use": 0,
             "vram_used_gb": 0.0, "vram_total_gb": 0.0,
             "gtt_total_gb": gtt_total, "gtt_used_gb": gtt_used,
             "temp_c": None, "power_w": None, "sclk": None}
        for line in out.splitlines():
            if f"GPU[{idx}]" not in line:
                continue
            m = re.search(r"GPU use \(%\):\s*(\d+)", line)
            if m: g["use"] = int(m.group(1))
            m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", line)
            if m: g["vram_total_gb"] = round(int(m.group(1)) / 1e9, 1)
            m = re.search(r"VRAM Total Used Memory \(B\):\s*(\d+)", line)
            if m: g["vram_used_gb"] = round(int(m.group(1)) / 1e9, 2)
            m = re.search(r"Temperature \(Sensor edge\) \(C\):\s*(\d+)", line)
            if m: g["temp_c"] = int(m.group(1))
            m = re.search(r"Average Graphics Package Power \(W\):\s*([\d.]+)", line)
            if m: g["power_w"] = float(m.group(1))
            m = re.search(r"Graphics Clock \(\d+ Hz\):\s*(\d+)", line)
            if m: g["sclk"] = int(m.group(1)) // 1000
        gpus.append(g)
    return gpus


def _collect_cpu(run):
    cpu = {"load1": None, "mem_used_gb": None, "mem_total_gb": None}
    out = run("cat /proc/loadavg; free -b", 10)
    lines = out.splitlines()
    if lines:
        m = re.match(r"([\d.]+)", lines[0].strip())
        if m: cpu["load1"] = float(m.group(1))
    for line in lines:
        if line.startswith("Mem:"):
            parts = line.split()
            if len(parts) > 2:
                cpu["mem_total_gb"] = round(int(parts[1]) / 1e9, 1)
                cpu["mem_used_gb"] = round(int(parts[2]) / 1e9, 1)
    return cpu


def local_gpu_mem_config():
    # Local-node GPU memory detail for the "Configure & Control" section.
    # Shows each GPU's rocm-smi VRAM, sysfs frame-buffer, and GTT (system RAM) pool,
    # plus the amdgpu.gttsize kernel lever for iGPUs.
    run = _run
    idxs, names = _gpu_indices(run)
    gpus = []
    for idx in idxs:
        g = {"index": idx, "name": names.get(idx, f"GPU{idx}"),
             "vram_total_gb": 0.0, "fb_total_gb": None, "gtt_total_gb": None,
             "gtt_used_gb": None, "is_igpu": False}
        # rocm-smi VRAM total
        out = run("rocm-smi --showmeminfo vram 2>/dev/null", 10)
        for line in out.splitlines():
            if f"GPU[{idx}]" in line:
                m = re.search(r"VRAM Total Memory \(B\):\s*(\d+)", line)
                if m: g["vram_total_gb"] = round(int(m.group(1)) / 1e9, 1)
        # sysfs framebuffer + GTT
        fb = run(f"cat /sys/class/drm/card{idx}/device/mem_info_vram_total 2>/dev/null", 5).strip()
        if fb.isdigit():
            g["fb_total_gb"] = round(int(fb) / 1e9, 1)
        gtt_total, gtt_used = _gtt_for_card(run, f"card{idx}")
        g["gtt_total_gb"] = gtt_total
        g["gtt_used_gb"] = gtt_used
        # iGPU heuristic: framebuffer (sysfs) is large (>=8GB) while rocm-smi VRAM is small
        # (the BIOS UMA display allocation, ~2GB). That gap means it pulls from GTT RAM.
        if g["fb_total_gb"] and g["fb_total_gb"] >= 8.0 and g["vram_total_gb"] and g["vram_total_gb"] < 4.0:
            g["is_igpu"] = True
        gpus.append(g)
    # amdgpu GTT kernel lever (empty = auto-sized by the driver)
    gttsize = run("cat /sys/module/amdgpu/parameters/gttsize 2>/dev/null", 5).strip() or "(auto)"
    vm_size = run("cat /sys/module/amdgpu/parameters/vm_size 2>/dev/null", 5).strip() or "(auto)"
    mem = run("grep MemTotal /proc/meminfo 2>/dev/null", 5)
    total_ram_gb = None
    m = re.search(r"MemTotal:\s*(\d+)", mem)
    if m:
        total_ram_gb = round(int(m.group(1)) / 1048576, 1)
    return {"gpus": gpus, "gttsize": gttsize, "vm_size": vm_size,
            "total_ram_gb": total_ram_gb}


def parse_hardware():
    nodes = []
    for node in load_nodes():
        run = node_run(node)
        idxs, names = _gpu_indices(run)
        gpus = _collect_gpus(run, idxs, names, gtt=(not node.get("remote")))
        nodes.append({"key": node["key"], "name": node["name"], "ip": node["ip"],
                      "role": node.get("role", ""), "reachable": bool(gpus or names),
                      "gpus": gpus, "cpu": _collect_cpu(run)})
    return {"nodes": nodes, "ts": int(time.time())}


# --- Rendezvous barrier state (written by hierarchos-vulkan-rendezvous) -----
RDZ_STATE = os.path.join(HOME, "hierarchos-rdz", "state.json")


def parse_rendezvous():
    info = {"running": False, "state": None}
    out = _run('pgrep -f "[h]ierarchos-vulkan-rendezvous" >/dev/null 2>&1 && echo 1 || echo 0', 5)
    info["running"] = out.strip() == "1"
    if os.path.exists(RDZ_STATE):
        try:
            with open(RDZ_STATE) as f:
                info["state"] = json.load(f)
            info["state"]["file_age_seconds"] = int(time.time() - os.path.getmtime(RDZ_STATE))
        except Exception as e:
            info["error"] = str(e)
    return info


# --- RDMA NIC hardware counters (live transfer rates) ------------------------
# Standard network monitors show 0 during RDMA because RoCE copies bypass the
# kernel stack. We read the NIC's own counters instead (mlx5 vport RDMA byte
# counters when available, else sysfs port counters).
_rdma_tracker = {}  # node_key -> {"tx":int, "rx":int, "ts":float}


def _read_rdma_counters(run, dev):
    if not dev:
        return None
    netdev = run(f"ls /sys/class/infiniband/{dev}/device/net/ 2>/dev/null | head -1", 5).strip()
    out = run(f"ethtool -S {netdev} 2>/dev/null", 8)
    tx = rx = None
    for line in out.splitlines():
        if "tx_vport_rdma_unicast_bytes" in line:
            tx = int(line.split()[-1])
        elif "rx_vport_rdma_unicast_bytes" in line:
            rx = int(line.split()[-1])
    if tx is not None and rx is not None:
        return {"tx": tx, "rx": rx, "source": "mlx5-vport", "netdev": netdev}
    sysfs = f"/sys/class/infiniband/{dev}/ports/1/counters"
    out2 = run(f"cat {sysfs}/port_xmit_data {sysfs}/port_rcv_data 2>/dev/null", 5)
    parts = out2.split()
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return {"tx": int(parts[0]) * 4, "rx": int(parts[1]) * 4, "source": "sysfs-words", "netdev": netdev}
    return None


def parse_rdma():
    nodes = []
    now = time.time()
    for node in load_nodes():
        key = node["key"]
        dev = node.get("rdma_device", "")
        run = node_run(node)
        counters = _read_rdma_counters(run, dev)
        info = {"key": key, "name": node["name"], "ip": node["ip"], "rdma_device": dev,
                "tx_mibps": 0.0, "rx_mibps": 0.0, "tx_gbps": 0.0, "rx_gbps": 0.0,
                "total_tx_gib": 0.0, "total_rx_gib": 0.0, "source": None, "netdev": None}
        if counters:
            info["source"] = counters["source"]
            info["netdev"] = counters["netdev"]
            info["total_tx_gib"] = round(counters["tx"] / 1e12, 2)
            info["total_rx_gib"] = round(counters["rx"] / 1e12, 2)
            prev = _rdma_tracker.get(key)
            if prev and now > prev["ts"]:
                dt = now - prev["ts"]
                dtx = max(0, counters["tx"] - prev["tx"])
                drx = max(0, counters["rx"] - prev["rx"])
                info["tx_mibps"] = round(dtx / dt / 1e6, 2)
                info["rx_mibps"] = round(drx / dt / 1e6, 2)
                info["tx_gbps"] = round(dtx * 8 / dt / 1e9, 2)
                info["rx_gbps"] = round(drx * 8 / dt / 1e9, 2)
            _rdma_tracker[key] = {"tx": counters["tx"], "rx": counters["rx"], "ts": now}
        nodes.append(info)
    return {"nodes": nodes, "ts": int(now)}


def tail_log(remote=None, n=40, key=None):
    content = _read_log(remote, key)
    if not content.strip():
        return "(no log)"
    return "".join(content.splitlines(keepends=True)[-n:])


def start_training(cfg):
    if training_running():
        return {"ok": False, "error": "Training already running locally. Stop it first."}
    epochs = int(cfg.get("epochs", 12))
    batch = int(cfg.get("batch_size", 64))
    lr = float(cfg.get("learning_rate", 1e-4))
    if cfg.get("devices"):
        device_list = [int(d) for d in cfg["devices"]]
    else:
        device_list = [int(cfg.get("device", 0))]
    if not device_list:
        device_list = [0]
    device_arg = ("--device-indices " + ",".join(map(str, device_list))) if len(device_list) > 1 else f"--device-index {device_list[0]}"
    precision = ("--precision fp16-storage-fp32-compute" if cfg.get("fp16", True) else "--precision fp32")
    ponder = float(cfg.get("ponder_weight", 0.0))
    data = cfg.get("data", "data/corpus_lm.jsonl")
    model_name = cfg.get("model_name", "out-engram").strip() or "out-engram"
    seed = int(cfg.get("seed", 1337))
    tokenizer = cfg.get("tokenizer", "openai-community/gpt2")
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", model_name):
        return {"ok": False, "error": "model_name may only contain letters, digits, _ - ."}
    outdir = os.path.join(WORK, model_name)
    cmd = (
        f"source ~/.cargo/env 2>/dev/null; cd {WORK} && rm -rf {outdir} .{outdir}* 2>/dev/null; "
        f"nohup {CLI} train "
        f"--hf-tokenizer {tokenizer} "
        f"--train {data} "
        f"--out-dir {model_name} "
        f"--epochs {epochs} --batch-size {batch} --lr {lr} "
        f"{device_arg} "
        f"--ponder-loss-weight {ponder} {precision} --seed {seed} "
        f"> {TRAIN_LOG} 2>&1 & echo LAUNCHED_PID_$!"
    )
    out = _run(cmd, timeout=20)
    m = re.search(r"LAUNCHED_PID_(\d+)", out)
    if m:
        _epoch_tracker.clear()
        return {"ok": True, "pid": int(m.group(1)), "model": model_name, "log": TRAIN_LOG}
    return {"ok": False, "error": f"Failed to launch: {out[-300:]}"}


def stop_training():
    nodes = load_nodes()
    for node in nodes:
        _ssh_run(node.get("remote"), 'pkill -f "[h]ierarchos-vulkan-train"; pkill -f "[h]ierarchos-native-cli train"', 15)
    time.sleep(1)
    return {"ok": True, "still_running": any(training_running(n.get("remote")) for n in nodes)}


def add_node(cfg):
    nodes = load_nodes()
    key = re.sub(r"[^a-z0-9_-]", "", (cfg.get("key") or cfg.get("name") or "").lower())[:24]
    if not key:
        return {"ok": False, "error": "name/key required"}
    if any(n["key"] == key for n in nodes):
        return {"ok": False, "error": f"node key '{key}' already exists"}
    ip = cfg.get("ip", "").strip()
    if not ip:
        return {"ok": False, "error": "ip required"}
    remote = (cfg.get("remote") or "").strip() or None
    # default ssh target if only an ip given
    if not remote and ip.count(".") == 3:
        remote = f"lunarbuntu@{ip}"
    nodes.append({"key": key, "name": cfg.get("name") or ip, "ip": ip,
                  "remote": remote, "role": cfg.get("role", "worker")})
    save_nodes(nodes)
    return {"ok": True, "nodes": nodes}


def remove_node(key):
    nodes = load_nodes()
    remaining = [n for n in nodes if n["key"] != key]
    if len(remaining) == len(nodes):
        return {"ok": False, "error": f"no node with key '{key}'"}
    save_nodes(remaining)
    return {"ok": True, "nodes": remaining}


def update_node(cfg):
    nodes = load_nodes()
    key = cfg.get("key", "")
    for n in nodes:
        if n["key"] == key:
            for field in ("dev_index", "rdma_device", "name", "role"):
                if field in cfg and cfg[field] is not None:
                    n[field] = cfg[field]
            save_nodes(nodes)
            return {"ok": True, "nodes": nodes}
    return {"ok": False, "error": f"no node with key '{key}'"}


RDZ_BIN = os.path.join(HOME, "Hierarchos-Native/hierarchos-vulkan/target/release/hierarchos-vulkan-rendezvous")
RDZ_PORT = 18599


def _launch_rank(node, rank, world_size, peer_ips, rendezvous, dist_cfg):
    """Launch one training rank on a node (local or over SSH).

    peer_ips: comma list rank->IP (rank r at index r). rendezvous: HOST:PORT barrier.
    Log -> ~/hierarchos-logs/<node_key>.log so the dashboard shows it live.
    """
    model_name = dist_cfg.get("model_name", "out-engram").strip() or "out-engram"
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", model_name):
        return {"ok": False, "error": "model_name may only contain letters, digits, _ - ."}
    outdir = f"{model_name}-rank{rank}"
    data = dist_cfg.get("data", "data/corpus_lm.jsonl")
    rdma = (node.get("rdma_device") or "").strip()
    if not rdma:
        return {"ok": False, "error": f"node {node['key']} has no rdma_device set"}
    dev_index = int(node.get("dev_index", 0))
    precision = ("--precision fp16-storage-fp32-compute" if dist_cfg.get("fp16", True)
                 else "--precision fp32")
    logf = os.path.join(HOME, "hierarchos-logs", f"{node['key']}.log")
    cmd = (
        f"source ~/.cargo/env 2>/dev/null; cd {WORK} && rm -rf {outdir} .{outdir}* 2>/dev/null; "
        f"mkdir -p {os.path.dirname(logf)}; "
        f"nohup {BIN} "
        f"--model {WORK}/{model_name} "
        f"--dataset {WORK}/{data} "
        f"--output {WORK}/{outdir} "
        f"--epochs {int(dist_cfg.get('epochs', 1))} --batch-size {int(dist_cfg.get('batch_size', 1))} "
        f"--gradient-accumulation-steps {int(dist_cfg.get('gas', 1))} "
        f"--lr {float(dist_cfg.get('learning_rate', 1e-4))} "
        f"--device-index {dev_index} --seed {int(dist_cfg.get('seed', 1337))} "
        f"--ponder-loss-weight {float(dist_cfg.get('ponder_weight', 0.0))} {precision} "
        f"{_offload_args(dist_cfg)} "
        f"--distributed-mode {dist_cfg.get('mode', 'allreduce')} "
        f"--rank {rank} --world-size {world_size} --peer-ips {peer_ips} --rdma-device {rdma} "
        f"--rendezvous {rendezvous} "
        f"> {logf} 2>&1 & echo LAUNCHED_PID_$!"
    )
    out = _ssh_run(node.get("remote"), cmd, timeout=30)
    m = re.search(r"LAUNCHED_PID_(\d+)", out)
    if m:
        return {"ok": True, "pid": int(m.group(1))}
    return {"ok": False, "error": out[-200:]}


def _start_rendezvous(world_size, rank0_node):
    """Start the rendezvous barrier on the RANK-0 host (fully detached).

    The helper is written + executed on rank-0's machine (locally, or via SSH if
    rank-0 is remote) so the barrier listens where the ranks expect it.
    """
    helper = os.path.join(HOME, "hierarchos-rdz", "start_rdz.sh")
    state = os.path.join(HOME, "hierarchos-rdz", "state.json")
    body = (
        "#!/bin/bash\n"
        "pkill -9 -f \"[h]ierarchos-vulkan-rendezvous\" 2>/dev/null; sleep 1\n"
        f"rm -f {state}\n"
        f"mkdir -p {os.path.dirname(helper)}\n"
        f"cd {WORK}\n"
        f"setsid nohup {RDZ_BIN} --listen 0.0.0.0:{RDZ_PORT} --world-size {world_size} "
        f"--state-file {state} </dev/null >/tmp/rdz.log 2>&1 &\n"
        "disown\n"
        "sleep 2\n"
        "echo RDZ_STARTED\n"
    )
    script = f"cat > {helper} << 'H'\n{body}H\nchmod +x {helper} && bash {helper}"
    if rank0_node.get("remote"):
        out = _ssh_run(rank0_node["remote"], script, timeout=30)
    else:
        out = _run(script, timeout=30)
    return "RDZ_STARTED" in out


def start_distributed(cfg):
    """Launch an N-rank distributed run across the selected registered nodes.

    cfg['node_keys']: list of node keys (default = every node with a 'rank').
    Ranks are taken from each node's stored 'rank'; the ring is wired via
    --peer-ips (rank r's IP at index r) plus a rendezvous barrier.
    """
    nodes = {n["key"]: n for n in load_nodes()}
    keys = cfg.get("node_keys")
    if not keys:
        keys = [n["key"] for n in load_nodes() if n.get("rank") is not None]
    sel = [nodes[k] for k in keys if k in nodes]
    if len(sel) < 2:
        return {"ok": False, "error": "select at least two nodes"}
    for n in sel:
        if training_running(n.get("remote")):
            return {"ok": False, "error": f"training already running on {n['key']}; stop it first"}
    sel = sorted(sel, key=lambda n: int(n.get("rank", 0)))
    world = len(sel)
    if int(cfg.get("gas", 1)) < world:
        return {"ok": False, "error": f"gas ({cfg.get('gas',1)}) must be >= world_size ({world})"}
    peer_ips = ",".join(n["ip"] for n in sel)   # rank r's IP at index r
    rendezvous = f"{sel[0]['ip']}:{RDZ_PORT}"     # barrier on the rank-0 host
    if not _start_rendezvous(world, sel[0]):
        return {"ok": False, "error": "failed to start rendezvous server"}
    pids = []
    for n in sel:
        rank = int(n.get("rank", 0))
        r = _launch_rank(n, rank, world, peer_ips, rendezvous, cfg)
        if not r["ok"]:
            return {"ok": False, "error": f"rank {rank} ({n['key']}): {r['error']}"}
        pids.append(r["pid"])
    _epoch_tracker.clear()
    return {"ok": True, "pids": pids, "world_size": world, "rendezvous": rendezvous}


def _launch_transformer_rank(node, rank, world_size, peer_ips, tcfg):
    """Launch one HuggingFace transformer rank on a node (local or SSH).

    The transformer distributed path uses the RDMA ring directly (no rendezvous
    daemon). Each rank gets its own output dir and (via the binary) its own
    per-rank SSD mmap file.
    """
    model = (tcfg.get("transformer_model") or "").strip()
    model_arg = _transformer_model_arg(model)
    if not model_arg:
        return {"ok": False, "error": "a transformer model path or HF hub id is required"}
    dataset = (tcfg.get("transformer_dataset") or "").strip() or "data/corpus_lm.jsonl"
    seq_len = int(tcfg.get("seq_len", 512))
    batch = int(tcfg.get("batch_size", 1))
    lr = float(tcfg.get("learning_rate", 1e-5))
    seed = int(tcfg.get("seed", 42))
    epochs = int(tcfg.get("epochs", 1))
    out = (tcfg.get("transformer_output") or "transformer-dist").strip() or "transformer-dist"
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", out):
        return {"ok": False, "error": "output name may only contain letters, digits, _ - ."}
    rdma = (node.get("rdma_device") or "").strip()
    if not rdma:
        return {"ok": False, "error": f"node {node['key']} has no rdma_device set"}
    dev_index = int(node.get("dev_index", 0))
    outdir = f"{out}-rank{rank}"
    logf = os.path.join(HOME, "hierarchos-logs", f"{node['key']}.log")
    cmd = (
        f"source ~/.cargo/env 2>/dev/null; cd {WORK} && rm -rf {outdir} .{outdir}* 2>/dev/null; "
        f"mkdir -p {os.path.dirname(logf)}; "
        f"nohup {BIN_TRANSFORMER} "
        f"{model_arg} "
        f"--dataset {dataset} "
        f"--output {outdir} "
        f"--epochs {epochs} --batch-size {batch} --seq-len {seq_len} "
        f"--learning-rate {lr} --seed {seed} "
        f"--device-index {dev_index} "
        f"{_offload_args(tcfg)} "
        f"--rank {rank} --world-size {world_size} --peer-ips {peer_ips} --rdma-device {rdma} "
        f"> {logf} 2>&1 & echo LAUNCHED_PID_$!"
    )
    outp = _ssh_run(node.get("remote"), cmd, timeout=30)
    m = re.search(r"LAUNCHED_PID_(\d+)", outp)
    if m:
        return {"ok": True, "pid": int(m.group(1))}
    return {"ok": False, "error": outp[-200:]}


def start_transformer_dist(cfg):
    """Launch a multi-rank HuggingFace transformer AllReduce run across nodes."""
    nodes = {n["key"]: n for n in load_nodes()}
    keys = cfg.get("node_keys")
    if not keys:
        keys = [n["key"] for n in load_nodes() if n.get("rank") is not None]
    sel = [nodes[k] for k in keys if k in nodes]
    if len(sel) < 2:
        return {"ok": False, "error": "select at least two nodes"}
    for n in sel:
        if training_running(n.get("remote")):
            return {"ok": False, "error": f"training already running on {n['key']}; stop it first"}
    sel = sorted(sel, key=lambda n: int(n.get("rank", 0)))
    world = len(sel)
    if not cfg.get("cpu_offload"):
        return {"ok": False, "error": "distributed transformer training requires CPU offload (enable it in the config)"}
    peer_ips = ",".join(n["ip"] for n in sel)   # rank r's IP at index r
    pids = []
    for n in sel:
        rank = int(n.get("rank", 0))
        r = _launch_transformer_rank(n, rank, world, peer_ips, cfg)
        if not r["ok"]:
            return {"ok": False, "error": f"rank {rank} ({n['key']}): {r['error']}"}
        pids.append(r["pid"])
    _epoch_tracker.clear()
    return {"ok": True, "pids": pids, "world_size": world, "peer_ips": peer_ips}


BIN = os.path.join(HOME, "Hierarchos-Native/hierarchos-vulkan/target/release/hierarchos-vulkan-train")
BIN_TRANSFORMER = os.path.join(HOME, "Hierarchos-Native/hierarchos-vulkan/target/release/hierarchos-vulkan-transformer-train")


def _offload_args(cfg):
    """Return the offload CLI fragment shared by both training variants."""
    if not cfg.get("cpu_offload"):
        return ""
    frag = "--cpu-offload"
    ssd = (cfg.get("offload_ssd") or "").strip()
    if ssd:
        cap = int(cfg.get("offload_ssd_capacity", 8589934592))
        frag += f" --offload-ssd {ssd} --offload-ssd-capacity-bytes {cap}"
    return frag


def _transformer_model_arg(model):
    """Decide whether the transformer model input is a local path or an HF hub id."""
    if not model:
        return None
    # Local path (absolute, relative that exists, or .safetensors/.json).json dir).
    if model.startswith("/") or os.path.exists(model) or model.endswith((".safetensors", ".json")):
        return f"--model {model}"
    return f"--hf-model {model}"


def start_transformer(cfg):
    """Launch a single-device HuggingFace transformer training run."""
    if training_running():
        return {"ok": False, "error": "Training already running locally. Stop it first."}
    model = (cfg.get("transformer_model") or "").strip()
    model_arg = _transformer_model_arg(model)
    if not model_arg:
        return {"ok": False, "error": "a transformer model path or HF hub id is required"}
    dataset = (cfg.get("transformer_dataset") or "").strip() or "data/corpus_lm.jsonl"
    seq_len = int(cfg.get("seq_len", 512))
    batch = int(cfg.get("batch_size", 1))
    lr = float(cfg.get("learning_rate", 1e-5))
    seed = int(cfg.get("seed", 42))
    epochs = int(cfg.get("epochs", 1))
    max_steps = cfg.get("max_steps", "")
    out = (cfg.get("transformer_output") or "transformer-run").strip() or "transformer-run"
    if not re.match(r"^[A-Za-z0-9_\-\.]+$", out):
        return {"ok": False, "error": "output name may only contain letters, digits, _ - ."}
    devices = [int(d) for d in (cfg.get("devices") or [0])]
    if len(devices) > 1:
        return {"ok": False, "error": "single-device transformer training uses one GPU; use the distributed launcher for multiple"}
    device_arg = f"--device-index {devices[0]}"
    text_field = (cfg.get("text_field") or "").strip()
    text_arg = f"--text-field {text_field}" if text_field else ""
    cmd = (
        f"source ~/.cargo/env 2>/dev/null; cd {WORK} && rm -rf {out} .{out}* 2>/dev/null; "
        f"nohup {BIN_TRANSFORMER} "
        f"{model_arg} {text_arg} "
        f"--dataset {dataset} "
        f"--output {out} "
        f"--epochs {epochs} --batch-size {batch} --seq-len {seq_len} "
        f"--learning-rate {lr} --seed {seed} "
        f"{device_arg} "
        f"{_offload_args(cfg)} "
    )
    if max_steps not in ("", None):
        cmd += f"--max-steps {int(max_steps)} "
    cmd += f"> {TRAIN_LOG} 2>&1 & echo LAUNCHED_PID_$!"
    outp = _run(cmd, timeout=20)
    m = re.search(r"LAUNCHED_PID_(\d+)", outp)
    if m:
        _epoch_tracker.clear()
        return {"ok": True, "pid": int(m.group(1)), "model": out, "log": TRAIN_LOG}
    return {"ok": False, "error": f"Failed to launch: {outp[-300:]}"}


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/":
            self._send(200, HTML, "text/html; charset=utf-8")
        elif path == "/api/status":
            self._send(200, parse_training())
        elif path == "/api/hardware":
            self._send(200, parse_hardware())
        elif path == "/api/rendezvous":
            self._send(200, parse_rendezvous())
        elif path == "/api/rdma":
            self._send(200, parse_rdma())
        elif path == "/api/nodes":
            self._send(200, {"nodes": load_nodes()})
        elif path == "/api/gpu-mem-config":
            self._send(200, local_gpu_mem_config())
        elif path == "/api/log":
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            n = int(q.get("lines", ["40"])[0])
            key = q.get("node", [""])[0]
            node = next((x for x in load_nodes() if x["key"] == key), None)
            self._send(200, {"log": tail_log(node.get("remote") if node else None, n, key)})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else "{}"
        try:
            cfg = json.loads(body)
        except Exception:
            cfg = {}
        if path == "/api/train":
            self._send(200, start_training(cfg))
        elif path == "/api/train-transformer":
            self._send(200, start_transformer(cfg))
        elif path == "/api/stop":
            self._send(200, stop_training())
        elif path == "/api/nodes/add":
            self._send(200, add_node(cfg))
        elif path == "/api/nodes/remove":
            self._send(200, remove_node(cfg.get("key", "")))
        elif path == "/api/nodes/update":
            self._send(200, update_node(cfg))
        elif path == "/api/train-distributed":
            self._send(200, start_distributed(cfg))
        elif path == "/api/train-distributed-transformer":
            self._send(200, start_transformer_dist(cfg))
        else:
            self._send(404, {"error": "not found"})


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Hierarchos Engram Trainer</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --border:#30363d; --txt:#e6edf3; --dim:#8b949e;
          --green:#3fb950; --yellow:#d29922; --red:#f85149; --blue:#58a6ff; --accent:#bc8cff; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
         background:var(--bg); color:var(--txt); }
  .wrap { max-width:1500px; margin:0 auto; padding:16px 20px; }
  h1 { font-size:22px; margin:0 0 4px; }
  .sub { color:var(--dim); font-size:13px; margin-bottom:16px; }
  /* top row: 35% hardware / 65% training */
  .toprow { display:grid; grid-template-columns:35fr 65fr; gap:16px; align-items:start; }
  @media (max-width:1100px){ .toprow{grid-template-columns:1fr;} }
  .panel { background:var(--panel); border:1px solid var(--border); border-radius:10px; padding:14px; }
  .panel h2 { font-size:14px; margin:0 0 10px; color:var(--dim); text-transform:uppercase; letter-spacing:.5px; }
  .train-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
  @media (max-width:800px){ .train-grid{grid-template-columns:1fr;} }
  .train-card { border:1px solid var(--border); border-radius:8px; padding:12px; background:#0d1117; }
  .train-card .card-title { font-size:14px; font-weight:700; color:var(--accent); margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }
  .node-label { font-size:11px; font-weight:700; color:var(--dim); text-transform:uppercase; letter-spacing:.5px; margin:10px 0 6px; display:flex; justify-content:space-between; align-items:center; }
  .node-label:first-child { margin-top:0; }
  .gpu-row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  @media (max-width:1300px){ .gpu-row{grid-template-columns:1fr;} }
  .gpu-card { border:1px solid var(--border); border-radius:8px; padding:10px; background:#0d1117; }
  .gpu-card .ghead { display:flex; justify-content:space-between; align-items:center; font-size:12px; margin-bottom:6px; }
  .gpu-card .ghead .gname { font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:70%; }
  .gpu-card .ghead .guse { font-weight:700; color:var(--blue); }
  .bar { height:7px; background:#21262d; border-radius:4px; overflow:hidden; }
  .bar > div { height:100%; background:var(--blue); transition:width .5s; }
  .metrics { display:flex; gap:10px; flex-wrap:wrap; font-size:11px; color:var(--dim); margin-top:5px; }
  .metrics b { color:var(--txt); font-weight:600; }
  .badge { display:inline-block; padding:2px 9px; border-radius:12px; font-size:11px; font-weight:600; }
  .badge.run { background:rgba(63,185,80,.15); color:var(--green); border:1px solid var(--green); }
  .badge.idle { background:rgba(139,148,158,.15); color:var(--dim); border:1px solid var(--border); }
  .kv { display:flex; justify-content:space-between; font-size:13px; padding:2px 0; }
  .kv span:first-child { color:var(--dim); }
  .prog { font-size:14px; margin:6px 0; }
  .prog .big { font-size:22px; font-weight:700; color:var(--accent); }
  label { display:block; font-size:12px; margin:10px 0 4px; }
  input[type=number], input[type=text], select { width:100%; padding:6px 9px; background:#0d1117;
    border:1px solid var(--border); border-radius:6px; color:var(--txt); font-size:13px; }
  .row2 { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .chk { display:flex; align-items:center; gap:8px; margin:10px 0; font-size:13px; }
  button { cursor:pointer; border:none; border-radius:6px; padding:9px 14px; font-size:13px; font-weight:600; }
  .btn-start { background:var(--green); color:#0d1117; }
  .btn-stop { background:var(--red); color:#fff; }
  .btn-sm { padding:4px 10px; font-size:12px; }
  .btn-add { background:var(--blue); color:#0d1117; }
  .btn-del { background:rgba(248,81,73,.15); color:var(--red); border:1px solid var(--red); }
  .btns { display:flex; gap:10px; margin-top:14px; }
  pre { background:#0d1117; border:1px solid var(--border); border-radius:8px; padding:10px; font-size:11px;
        overflow:auto; max-height:260px; color:var(--dim); font-family:ui-monospace,Menlo,monospace; }
  .full { grid-column:1 / -1; }
  .grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:16px; }
  @media (max-width:900px){ .grid{grid-template-columns:1fr;} }
  .nodeform { display:grid; grid-template-columns:1fr 1fr 1fr auto; gap:8px; margin-top:8px; align-items:end; }
  /* logs: resizable side-by-side, wrapping text */
  #logs { display:flex; gap:16px; margin-top:16px; align-items:flex-start; }
  #logs > .panel { flex:1 1 0; min-width:0; resize:horizontal; overflow:auto; max-width:100%; }
  @media (max-width:900px){ #logs{flex-direction:column;} #logs>.panel{resize:none;} }
  #logs pre { white-space:pre-wrap; word-break:break-word; overflow-wrap:anywhere; }
  .node-table { width:100%; border-collapse:collapse; font-size:12px; }
  .node-table th { text-align:left; color:var(--dim); font-weight:600; padding:4px 8px; border-bottom:1px solid var(--border); }
  .node-table td { padding:4px 8px; border-bottom:1px solid #21262d; }
  .node-table input { width:110px; padding:4px 8px; font-size:12px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>🧠 Hierarchos Engram Trainer</h1>
  <div class="sub">Distributed multi-node monitor · auto-refresh 5s</div>

  <!-- RENDEZVOUS BARRIER OVERVIEW -->
  <div class="panel" style="margin-bottom:16px">
    <h2>Rendezvous Barrier</h2>
    <div id="rendezvous" style="font-size:13px;color:var(--dim)">loading…</div>
  </div>

  <!-- RDMA MONITOR -->
  <div class="panel" style="margin-bottom:16px">
    <h2>RDMA Monitor <span style="text-transform:none;font-weight:400">(NIC hardware counters)</span></h2>
    <div id="rdma" style="font-size:13px;color:var(--dim)">loading…</div>
  </div>

  <div class="toprow">
    <!-- HARDWARE (35%) -->
    <div class="panel">
      <h2>Hardware</h2>
      <div id="hw"></div>
    </div>

    <!-- TRAINING STATUS (65%) -->
    <div class="panel">
      <h2>Training Status</h2>
      <div class="train-grid" id="train"></div>
    </div>
  </div>

  <!-- NODES -->
  <div class="panel" style="margin-top:16px">
    <h2>Nodes</h2>
    <table class="node-table" id="nodes-table">
      <thead><tr><th>Node</th><th>IP</th><th>Role</th><th>Train GPU index</th><th>RDMA device</th><th></th></tr></thead>
      <tbody id="nodes"></tbody>
    </table>
    <div class="nodeform">
      <div><label>Name / key</label><input type="text" id="node-name" placeholder="pc3-worker"></div>
      <div><label>IP</label><input type="text" id="node-ip" placeholder="192.168.1.x"></div>
      <div><label>SSH target (blank = lunarbuntu@IP)</label><input type="text" id="node-remote" placeholder="user@host"></div>
      <button class="btn-add" onclick="addNode()">+ Add node</button>
    </div>
    <div id="node-msg" style="margin-top:6px;font-size:12px"></div>
  </div>

  <!-- DISTRIBUTED LAUNCH -->
  <div class="panel" style="margin-top:16px">
    <h2>Distributed Training Launch (N nodes)</h2>
    <div class="row2">
      <div style="grid-column:1 / -1">
        <label>GPUs / ranks (check the GPUs to include; ring order = rank from Nodes table)</label>
        <div id="dist-node-checks" style="margin-top:4px"></div>
      </div>
      <div>
        <label>Variant</label>
        <select id="dist-variant" onchange="toggleDistVariant()">
          <option value="hierarchos">Hierarchos (RWKV)</option>
          <option value="transformer">HF Transformer</option>
        </select>
      </div>
      <div id="dist-hier-fields">
        <label>Mode</label>
        <select id="dist-mode">
          <option value="allreduce">allreduce (every rank steps optimizer)</option>
          <option value="param-server">param-server (primary owns optimizer)</option>
        </select>
      </div>
      <div id="dist-hier-fields">
        <label>Gradient accumulation steps (must be ≥ #GPUs)</label>
        <input type="number" id="dist-gas" value="6" min="1" max="256">
      </div>
    </div>
    <div class="btns">
      <button class="btn-start" onclick="startDist()">▶ Launch Distributed</button>
    </div>
    <div id="dist-msg" style="margin-top:8px;font-size:12px"></div>
    <p style="font-size:11px;color:var(--dim);margin-top:8px">Reuses the shared fields below (model, data, epochs, batch, lr, seed, offload). Each node's GPU + RDMA device come from the Nodes table above. <b>HF Transformer distributed requires CPU/SSD offload enabled.</b> Logs appear in the log panels below.</p>
  </div>

  <!-- CONFIG -->
  <div class="panel" style="margin-top:16px">
    <h2>Configure &amp; Control (local)</h2>

    <!-- VARIANT SELECTOR -->
    <div style="margin-bottom:14px;padding:10px 12px;background:var(--bg2);border:1px solid var(--border);border-radius:8px;display:flex;gap:22px;align-items:center;flex-wrap:wrap">
      <span style="font-size:12px;font-weight:600;color:var(--dim);letter-spacing:.5px">TRAINING VARIANT</span>
      <label class="chk"><input type="radio" name="variant" value="hierarchos" checked onchange="toggleVariant()"> <b>Hierarchos (RWKV)</b></label>
      <label class="chk"><input type="radio" name="variant" value="transformer" onchange="toggleVariant()"> <b>HF Transformer</b> (Qwen / Llama / GPT-2 / …)</label>
    </div>

    <!-- SHARED FIELDS -->
    <div class="row2">
      <div>
        <label>Epochs</label>
        <input type="number" id="epochs" value="12" min="1" max="500">
      </div>
      <div>
        <label>Batch size</label>
        <input type="number" id="batch_size" value="64" min="1" max="1024">
      </div>
      <div>
        <label>Learning rate</label>
        <input type="number" id="learning_rate" value="0.0001" step="0.00001">
      </div>
      <div>
        <label>Random seed</label>
        <input type="number" id="seed" value="1337">
      </div>
      <div>
        <label>GPUs</label>
        <div class="chk" style="margin:6px 0"><input type="checkbox" class="gpu-chk" value="0" checked> <span><b>GPU 0</b> — RX 7600 XT</span></div>
        <div class="chk" style="margin:6px 0"><input type="checkbox" class="gpu-chk" value="1"> <span><b>GPU 1</b> — 8060S iGPU</span></div>
      </div>
      <div>
        <label>Output model name</label>
        <input type="text" id="model_name" value="out-engram">
      </div>
    </div>

    <!-- HIERARCHOS-ONLY FIELDS -->
    <div id="hier-fields">
      <div class="row2" style="margin-top:12px">
        <div><label>Tokenizer</label><input type="text" id="tokenizer" value="openai-community/gpt2"></div>
        <div><label>Training data</label><input type="text" id="data" value="data/corpus_lm.jsonl"></div>
        <div><label>Ponder loss weight</label><input type="number" id="ponder_weight" value="0" step="0.001" min="0"></div>
        <div class="chk"><input type="checkbox" id="fp16" checked> <span><b>fp16 mixed precision</b></span></div>
      </div>
    </div>

    <!-- TRANSFORMER-ONLY FIELDS -->
    <div id="transformer-fields" style="display:none;margin-top:12px">
      <div class="row2">
        <div><label>Model path or HF hub id</label><input type="text" id="transformer_model" placeholder="/path/to/model  or  Qwen/Qwen3-0.6B" value="Qwen/Qwen3-0.6B"></div>
        <div><label>Dataset (jsonl)</label><input type="text" id="transformer_dataset" value="data/corpus_lm.jsonl"></div>
        <div><label>Sequence length</label><input type="number" id="seq_len" value="512" min="1" max="8192"></div>
        <div><label>Max steps (0 = until data ends)</label><input type="number" id="max_steps" value="0" min="0"></div>
        <div><label>Text field name (default: text)</label><input type="text" id="text_field" value="text"></div>
      </div>
    </div>

    <!-- OFFLOAD (both variants) -->
    <div id="offload-panel" style="margin-top:12px;padding:10px;background:var(--bg2);border:1px solid var(--border);border-radius:8px">
      <div class="chk" style="margin-bottom:8px"><input type="checkbox" id="cpu_offload" onchange="toggleOffloadSsd()"> <span><b>CPU / SSD optimizer offload</b> (moves AdamW moments to host RAM or mmap'd SSD)</span></div>
      <div id="offload-ssd-row" class="row2" style="display:none">
        <div><label>SSD mmap file path</label><input type="text" id="offload_ssd" value="/tmp/offload.mmap"></div>
        <div><label>SSD capacity (bytes)</label><input type="number" id="offload_ssd_capacity" value="8589934592" min="0"></div>
      </div>
      <div style="font-size:11px;color:var(--dim);margin-top:6px">Leave the SSD path empty for a pure RAM (heap) store. Required for distributed HF transformer training.</div>
    </div>

    <!-- GPU MEMORY & iGPU GTT CONFIG -->
    <div style="margin-top:12px;padding:10px;background:var(--bg2);border:1px solid var(--border);border-radius:8px">
      <div style="font-size:12px;font-weight:600;margin-bottom:8px;color:var(--green)">GPU Memory &amp; iGPU GTT Configuration</div>
      <div id="gpu-mem-config"><div class="metrics">loading…</div></div>
      <div id="gtt-config-note" style="font-size:11px;color:var(--dim);margin-top:8px;line-height:1.5"></div>
    </div>

    <div class="btns" style="margin-top:12px">
      <button class="btn-start" onclick="startTrain()">▶ Start Training</button>
      <button class="btn-stop" onclick="stopTrain()">■ Stop (all nodes)</button>
    </div>
    <div id="action-msg" style="margin-top:10px;font-size:13px"></div>
  </div>

  <!-- LOSS CHART -->
  <div class="panel" style="margin-top:16px">
    <h2>Loss Curve — Primary</h2>
    <svg id="chart" width="100%" height="150" style="background:#0d1117;border:1px solid var(--border);border-radius:8px"></svg>
  </div>

  <!-- LOGS -->
  <div class="grid" id="logs"></div>
</div>

<script>
function fmtDur(s){ if(s==null) return "–"; s=Math.round(s); const h=Math.floor(s/3600),m=Math.floor(s%3600/60),ss=s%60;
  return (h?h+"h ":"")+(m?m+"m ":"")+ss+"s"; }
function bar(v,max){ return Math.min(100,Math.max(0,(v/max)*100)).toFixed(1)+"%"; }

async function loadGpuMemConfig(){
  try{
    const d = await (await fetch("/api/gpu-mem-config")).json();
    let html="";
    for(const g of d.gpus){
      const igpu = g.is_igpu ? ' <span style="color:#e3a008">[iGPU · GTT]</span>' : '';
      const fb = g.fb_total_gb ? `FB ${g.fb_total_gb}GB · ` : '';
      const gtt = g.gtt_total_gb ? `GTT ${g.gtt_total_gb}GB (used ${g.gtt_used_gb??'?'}GB)` : 'GTT n/a';
      html+=`<div class="metrics" style="margin-bottom:6px">
        <span><b>GPU ${g.index}</b> ${g.name}${igpu}</span>
        <span>VRAM <b>${g.vram_total_gb||'?'}GB</b></span>
        <span>${fb}${gtt}</span>
      </div>`;
    }
    document.getElementById("gpu-mem-config").innerHTML = html;
    const igpu = d.gpus.find(g=>g.is_igpu);
    let note = `Host RAM: ${d.total_ram_gb??'?'}GB · amdgpu.gttsize=${d.gttsize} · vm_size=${d.vm_size}`;
    if(igpu){
      note += `<br>iGPU (GPU ${igpu.index}) pulls ${igpu.fb_total_gb}GB from GTT system RAM (rocm-smi shows only ${igpu.vram_total_gb}GB display-UMA). The Vulkan budget exposes ${igpu.fb_total_gb}GB — usable as-is. To raise the ceiling, set kernel param <code>amdgpu.gttsize=&lt;KB&gt;</code> and reboot; more host RAM = more headroom.`;
    }
    document.getElementById("gtt-config-note").innerHTML = note;
  }catch(e){ document.getElementById("gpu-mem-config").innerHTML="<div class='metrics'>gpu-mem error: "+e+"</div>"; }
}

async function loadHardware(){
  try{
    const d = await (await fetch("/api/hardware")).json();
    let html="";
    for(const n of d.nodes){
      html+=`<div class="node-label"><span>${n.name}</span><span style="color:var(--dim)">${n.ip}</span></div>`;
      if(!n.reachable){ html+=`<div class="metrics">unreachable</div>`; continue; }
      html+=`<div class="gpu-row">`;
      for(const g of n.gpus){
        html+=`<div class="gpu-card">
          <div class="ghead"><span class="gname" title="${g.name}">${g.name}</span><span class="guse">${g.use}%</span></div>
          <div class="bar"><div style="width:${g.use}%;background:${g.use>50?'var(--green)':'var(--blue)'}"></div></div>
          <div class="metrics">
            <span>VRAM <b>${g.vram_used_gb}/${g.vram_total_gb||'?'}GB</b></span>
            <span>Temp <b>${g.temp_c??'–'}°C</b></span>
            <span>Power <b>${g.power_w?g.power_w.toFixed(0)+'W':'–'}</b></span>
          </div>
        </div>`;
      }
      html+=`</div>`;
      const c=n.cpu;
      html+=`<div class="metrics" style="margin-top:8px">
        <span>CPU load <b>${c.load1??'–'}</b></span>
        <span>RAM <b>${c.mem_used_gb??'–'}/${c.mem_total_gb??'–'}GB</b></span></div>`;
    }
    document.getElementById("hw").innerHTML = html;
  }catch(e){ document.getElementById("hw").innerHTML="<div class='metrics'>hardware error: "+e+"</div>"; }
}

async function loadRendezvous(){
  try{
    const d = await (await fetch("/api/rendezvous")).json();
    const el = document.getElementById("rendezvous");
    if(!d.state){
      el.innerHTML = d.running ? '<span style="color:var(--yellow)">rendezvous server running, no state file yet</span>'
        : '<span>No rendezvous state found. Start <code>hierarchos-vulkan-rendezvous --listen 0.0.0.0:18599 --world-size N</code> (state written to ~/hierarchos-rdz/state.json).</span>';
      return;
    }
    const s = d.state;
    const statusColor = s.timed_out ? 'var(--red)' : (s.go_sent ? 'var(--green)' : 'var(--yellow)');
    const statusText = s.timed_out ? 'TIMED OUT' : (s.go_sent ? 'GO SENT ✓' : `WAITING (${s.checked_in}/${s.world_size})`);
    let html = `<span class="badge ${s.go_sent||s.timed_out?'':'run'}" style="border-color:${statusColor};color:${statusColor};background:transparent">${statusText}</span> `;
    html += `server <b>${d.running?'<span style="color:var(--green)">running</span>':'<span style="color:var(--red)">stopped</span>'}</b> · listen <b>${s.listen}</b> · world-size <b>${s.world_size}</b> · `;
    html += `state file <b>${s.file_age_seconds!=null? fmtDur(s.file_age_seconds)+' ago':'?'}</b>`;
    if(s.check_ins && s.check_ins.length){
      html += `<div style="margin-top:8px">`;
      for(const c of s.check_ins){
        html += `<div class="kv"><span>check-in</span><b style="color:var(--blue)">${c.addr}</b></div>`;
      }
      html += `</div>`;
    }
    el.innerHTML = html;
  }catch(e){ document.getElementById("rendezvous").innerHTML="rendezvous error: "+e; }
}

async function loadRdma(){
  try{
    const d = await (await fetch("/api/rdma")).json();
    let html="";
    for(const n of d.nodes){
      if(!n.source){
        html+=`<div class="node-label"><span>${n.name} <span style="color:var(--dim)">(${n.rdma_device||'no rdma_device'})</span></span></div><div class="metrics">no RDMA counters (ethtool/sysfs unavailable)</div>`;
        continue;
      }
      const active = (n.tx_gbps>0.01||n.rx_gbps>0.01);
      const dot = active? '<span style="color:var(--green)">●</span>': '<span style="color:var(--dim)">○</span>';
      html+=`<div class="node-label"><span>${dot} ${n.name} <span style="color:var(--dim)">(${n.rdma_device}${n.netdev?', '+n.netdev:''})</span></span><span style="color:var(--dim);font-size:11px">${n.source}</span></div>`;
      html+=`<div class="gpu-row">`;
      // TX bar (scale to 30 Gbps)
      const txpct=Math.min(100, n.tx_gbps/30*100).toFixed(1);
      const rxpct=Math.min(100, n.rx_gbps/30*100).toFixed(1);
      html+=`<div class="gpu-card"><div class="ghead"><span class="gname">TX</span><span class="guse">${n.tx_gbps.toFixed(2)} Gbps</span></div>
        <div class="bar"><div style="width:${txpct}%;background:var(--accent)"></div></div>
        <div class="metrics"><span>${n.tx_mibps} MiB/s</span><span>total <b>${n.total_tx_gib} GiB</b></span></div></div>`;
      html+=`<div class="gpu-card"><div class="ghead"><span class="gname">RX</span><span class="guse">${n.rx_gbps.toFixed(2)} Gbps</span></div>
        <div class="bar"><div style="width:${rxpct}%;background:var(--green)"></div></div>
        <div class="metrics"><span>${n.rx_mibps} MiB/s</span><span>total <b>${n.total_rx_gib} GiB</b></span></div></div>`;
      html+=`</div>`;
    }
    document.getElementById("rdma").innerHTML = html;
  }catch(e){ document.getElementById("rdma").innerHTML="rdma error: "+e; }
}

async function loadStatus(){
  try{
    const d = await (await fetch("/api/status")).json();
    const keys = Object.keys(d);
    let html="";
    for(const key of keys){
      const t = d[key];
      const badge = t.running? '<span class="badge run">training</span>' : '<span class="badge idle">idle</span>';
      html+=`<div class="train-card">
        <div class="card-title"><span>${key.charAt(0).toUpperCase()+key.slice(1)}</span>${badge}</div>
        <div class="kv"><span>Epoch</span><b>${t.total_epochs? `${t.current_epoch}/${t.total_epochs}` : '–'}</b></div>
        <div class="kv"><span>Optimizer step</span><b>${t.optimizer_step||'–'}</b></div>
        <div class="kv"><span>Last loss</span><b>${t.last_loss? t.last_loss.toFixed(3) : '–'}</b></div>
        <div class="kv"><span>LR</span><b>${t.live_lr? t.live_lr.toExponential(2) : '–'}</b></div>
        <div class="kv"><span>Elapsed / ETA</span><b>${fmtDur(t.elapsed_seconds)} / ${fmtDur(t.eta_seconds)}</b></div>
      </div>`;
      buildLogPanel(key);
    }
    document.getElementById("train").innerHTML = html;
    drawChart((d.primary&&d.primary.loss_history)||[]);
  }catch(e){}
}

function buildLogPanel(key){
  const logs=document.getElementById("logs");
  if(!logs.querySelector(`[data-node="${key}"]`)){
    const p=document.createElement("div"); p.setAttribute("data-node",key); p.className="panel";
    p.innerHTML=`<h2>Log — ${key}</h2><pre id="log-${key}">loading…</pre>`;
    logs.appendChild(p);
  }
}

async function loadLogs(keys){
  for(const key of keys){
    try{ const d=await (await fetch(`/api/log?lines=30&node=${key}`)).json();
      const el=document.getElementById("log-"+key); if(el){ el.textContent=d.log||"(no log)"; el.scrollTop=1e9; } }catch(e){}
  }
}

function drawChart(hist){
  const svg=document.getElementById("chart"); svg.innerHTML="";
  if(!hist.length){ const t=document.createElementNS("http://www.w3.org/2000/svg","text");
    t.setAttribute("x",20);t.setAttribute("y",80);t.setAttribute("fill","#8b949e");t.textContent="no epoch data yet";svg.appendChild(t); return; }
  const W=svg.clientWidth||900, H=150, pad=30;
  const xs=hist.map(h=>h.epoch), ys=hist.map(h=>h.loss);
  const minX=Math.min(...xs), maxX=Math.max(...xs), minY=Math.min(...ys), maxY=Math.max(...ys);
  const X=v=>pad+(v-minX)/((maxX-minX)||1)*(W-2*pad);
  const Y=v=>H-pad-(v-minY)/((maxY-minY)||1)*(H-2*pad);
  const pl=document.createElementNS("http://www.w3.org/2000/svg","polyline");
  pl.setAttribute("points",hist.map(h=>`${X(h.epoch)},${Y(h.loss)}`).join(" "));
  pl.setAttribute("fill","none");pl.setAttribute("stroke","#bc8cff");pl.setAttribute("stroke-width","2");
  svg.appendChild(pl);
  hist.forEach(h=>{const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
    c.setAttribute("cx",X(h.epoch));c.setAttribute("cy",Y(h.loss));c.setAttribute("r","3");c.setAttribute("fill","#bc8cff");
    svg.appendChild(c);});
}

async function loadNodes(){
  try{
    const d=await (await fetch("/api/nodes")).json();
    let html="";
    for(const n of d.nodes){
      html+=`<tr>
        <td>${n.name} <span style="color:var(--dim)">(${n.ip})</span></td>
        <td>${n.ip}</td>
        <td>${n.role||'worker'}</td>
        <td><input type="number" value="${n.dev_index??0}" min="0" max="15" onchange="updateNodeCfg('${n.key}','dev_index',this.value)"></td>
        <td><input type="text" value="${n.rdma_device||''}" placeholder="rocepXs0" onchange="updateNodeCfg('${n.key}','rdma_device',this.value)"></td>
        <td><button class="btn-sm btn-del" onclick="removeNode('${n.key}')">remove</button></td>
      </tr>`;
    }
    document.getElementById("nodes").innerHTML = html;
    // populate distributed launch checkboxes (one per GPU/rank)
    const box=document.getElementById("dist-node-checks");
    if(box){
      const cur=new Set([...box.querySelectorAll("input:checked")].map(c=>c.value));
      const opts=[...d.nodes].sort((a,b)=>(a.rank??99)-(b.rank??99)).map(n=>
        `<div class="chk" style="margin:3px 0"><label><input type="checkbox" value="${n.key}" ${(!cur.size||cur.has(n.key))?"checked":""}> 
          <b>rank ${n.rank??'?'}</b> — ${n.name} (GPU ${n.dev_index??0}, ${n.ip})</label></div>`).join("");
      if(box.dataset.count!=String(d.nodes.length)){ box.innerHTML=opts; box.dataset.count=String(d.nodes.length); }
    }
  }catch(e){}
}

async function updateNodeCfg(key, field, value){
  try{
    await fetch("/api/nodes/update",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({key, [field]: value})});
  }catch(e){}
}

async function startDist(){
  const checks=document.querySelectorAll("#dist-node-checks input[type=checkbox]");
  const keys=[...checks].filter(c=>c.checked).map(c=>c.value);
  if(keys.length<2){ alert("Select at least two GPUs"); return; }
  const variant=document.getElementById("dist-variant").value;
  const cfg={ node_keys:keys,
    epochs:+document.getElementById("epochs").value,
    batch_size:+document.getElementById("batch_size").value,
    learning_rate:+document.getElementById("learning_rate").value,
    seed:+document.getElementById("seed").value,
    model_name:document.getElementById("model_name").value,
    ...offloadCfg() };
  let endpoint="/api/train-distributed";
  if(variant==="transformer"){
    Object.assign(cfg, transformerCfg());
    endpoint="/api/train-distributed-transformer";
  } else {
    cfg.mode=document.getElementById("dist-mode").value;
    cfg.gas=+document.getElementById("dist-gas").value;
    cfg.data=document.getElementById("data").value;
    cfg.fp16=document.getElementById("fp16").checked;
  }
  const r=await (await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cfg)})).json();
  const m=document.getElementById("dist-msg");
  m.textContent=r.ok?`✓ Launched ${r.world_size} ranks (pids ${r.pids.join(", ")}). Watch the logs panel.`:("✗ "+r.error);
  m.style.color=r.ok?"var(--green)":"var(--red)";
  setTimeout(()=>{m.textContent="";},10000);
  loadStatus();
}

async function addNode(){
  const cfg={ name:document.getElementById("node-name").value,
    ip:document.getElementById("node-ip").value,
    remote:document.getElementById("node-remote").value };
  const r=await (await fetch("/api/nodes/add",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cfg)})).json();
  const m=document.getElementById("node-msg");
  m.textContent=r.ok?"✓ Node added. (SSH key auth must be set up for stats.)":("✗ "+r.error);
  m.style.color=r.ok?"var(--green)":"var(--red)";
  setTimeout(()=>{m.textContent="";},6000);
  loadNodes();
}
async function removeNode(key){
  if(!confirm("Remove node "+key+"?")) return;
  const r=await (await fetch("/api/nodes/remove",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key})})).json();
  loadNodes();
}

function currentVariant(){
  const el=document.querySelector('input[name=variant]:checked');
  return el ? el.value : "hierarchos";
}
function toggleVariant(){
  const v=currentVariant();
  document.getElementById("hier-fields").style.display = v==="hierarchos" ? "" : "none";
  document.getElementById("transformer-fields").style.display = v==="transformer" ? "" : "none";
}
function toggleOffloadSsd(){
  const on=document.getElementById("cpu_offload").checked;
  document.getElementById("offload-ssd-row").style.display = on ? "" : "none";
}
function toggleDistVariant(){
  const v=document.getElementById("dist-variant").value;
  document.querySelectorAll("#dist-hier-fields").forEach(el=>{ el.style.display = v==="hierarchos" ? "" : "none"; });
}
function offloadCfg(){
  return { cpu_offload: document.getElementById("cpu_offload").checked,
    offload_ssd: document.getElementById("offload_ssd").value,
    offload_ssd_capacity: +document.getElementById("offload_ssd_capacity").value };
}
function transformerCfg(){
  return { transformer_model: document.getElementById("transformer_model").value,
    transformer_dataset: document.getElementById("transformer_dataset").value,
    seq_len: +document.getElementById("seq_len").value,
    max_steps: +document.getElementById("max_steps").value,
    text_field: document.getElementById("text_field").value,
    transformer_output: document.getElementById("model_name").value };
}
async function startTrain(){
  const shared={ epochs:+document.getElementById("epochs").value,
    batch_size:+document.getElementById("batch_size").value,
    learning_rate:+document.getElementById("learning_rate").value,
    seed:+document.getElementById("seed").value,
    devices:[...document.querySelectorAll(".gpu-chk:checked")].map(c=>+c.value),
    model_name:document.getElementById("model_name").value };
  const cfg={ ...shared, ...offloadCfg() };
  let endpoint="/api/train";
  if(currentVariant()==="transformer"){
    Object.assign(cfg, transformerCfg());
    endpoint="/api/train-transformer";
  } else {
    cfg.tokenizer=document.getElementById("tokenizer").value;
    cfg.data=document.getElementById("data").value;
    cfg.fp16=document.getElementById("fp16").checked;
    cfg.ponder_weight=+document.getElementById("ponder_weight").value;
  }
  const r=await (await fetch(endpoint,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(cfg)})).json();
  const m=document.getElementById("action-msg");
  m.textContent = r.ok? `✓ Launched (pid ${r.pid})` : `✗ ${r.error}`;
  m.style.color = r.ok? "var(--green)" : "var(--red)";
  setTimeout(()=>{ m.textContent=""; },8000);
  loadStatus();
}
async function stopTrain(){
  const r=await (await fetch("/api/stop",{method:"POST"})).json();
  const m=document.getElementById("action-msg");
  m.textContent = r.still_running? "stopping…" : "✓ Stopped.";
  m.style.color = "var(--yellow)";
  loadStatus();
}

let _nodeKeys=[];
function tick(){
  loadRendezvous();
  loadRdma();
  loadHardware();
  loadStatus().then(()=>loadLogs(_nodeKeys));
  loadNodes();
}
async function firstTick(){
  try{ const d=await (await fetch("/api/status")).json(); _nodeKeys=Object.keys(d); }catch(e){}
  tick();
}
firstTick(); setInterval(tick, 5000);
// GPU memory config is slow (rocm-smi/sysfs per GPU) — poll less often
loadGpuMemConfig(); setInterval(()=>loadGpuMemConfig(), 20000);
</script>
</body>
</html>
"""


class ThreadingServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


if __name__ == "__main__":
    print(f"Hierarchos dashboard on http://0.0.0.0:{PORT}")
    with ThreadingServer(("0.0.0.0", PORT), Handler) as httpd:
        httpd.serve_forever()
