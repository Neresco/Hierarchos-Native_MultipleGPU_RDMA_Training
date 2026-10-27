# Hierarchos Dashboard (WebUI)

A dependency-free Python control UI for monitoring and launching distributed
GPU training across the cluster. It serves a single-page app on a local port
(default `7788`) that shows hardware, RDMA, rendezvous, and per-rank training
state, and can start/stop training runs.

Only the Python standard library is required (`http.server`, `socketserver`,
`json`, `subprocess`, `urllib`). No `pip install` step.

## Quick start

```bash
# from this directory (or any directory)
./dash_start.sh                 # start  -> http://<host>:7788/
./dash_start.sh status          # show pid + port
./dash_start.sh nodes           # list registered nodes (rank/GPU/RDMA)
./dash_start.sh restart         # reload server.py after edits
./dash_start.sh stop            # stop
```

Environment overrides:

- `DASH_PORT` — port to serve on (default `7788`).
- `DASH_HOST` — bind address (default `0.0.0.0`).

The server binds `0.0.0.0`, so it is reachable from other machines on the LAN
(e.g. `http://192.168.100.20:7788/` when running on PC2).

## Registering nodes (multi-GPU / multi-machine)

The dashboard launches ranks by **node**. Each node describes one GPU endpoint:
its key, display name, LAN IP, SSH target, ring rank, local GPU device index, and
RDMA device. Registered nodes live in `nodes.json` next to `server.py`.

- **Edit `nodes.json` directly**, then `./dash_start.sh restart`.
- **Or add/update from the browser** (Nodes panel): name/key, IP, SSH target
  (`user@host`, blank = local), GPU device index, RDMA device, and ring rank.

The ring order is determined by each node's `rank` field (rank 0..N-1). For an
N-rank run every rank must have a distinct `rank` and a working `rdma_device`.
Example 6-GPU cluster (4x RX 9060 XT on PC1 + RX 7600 XT / 8060S iGPU on PC2):

```json
[
  {"key": "pc1-gpu0", "ip": "192.168.100.101", "remote": "lunarbuntu@192.168.100.101", "rank": 0, "dev_index": 0, "rdma_device": "rocep7s0"},
  {"key": "pc1-gpu1", "ip": "192.168.100.101", "remote": "lunarbuntu@192.168.100.101", "rank": 1, "dev_index": 1, "rdma_device": "rocep7s0"},
  {"key": "pc1-gpu2", "ip": "192.168.100.101", "remote": "lunarbuntu@192.168.100.101", "rank": 2, "dev_index": 2, "rdma_device": "rocep7s0"},
  {"key": "pc1-gpu3", "ip": "192.168.100.101", "remote": "lunarbuntu@192.168.100.101", "rank": 3, "dev_index": 3, "rdma_device": "rocep7s0"},
  {"key": "pc2-gpu0", "ip": "192.168.100.20",  "remote": null,                              "rank": 4, "dev_index": 0, "rdma_device": "rocep198s0"},
  {"key": "pc2-gpu1", "ip": "192.168.100.20",  "remote": null,                              "rank": 5, "dev_index": 1, "rdma_device": "rocep198s0"}
]
```

`remote: null` means "run locally" (the dashboard host itself).

## Launching training

The **Configure & Control** panel has a **training-variant** selector:

- **HF Transformer** (`hierarchos-vulkan-transformer-train`): model path or HF
  hub id, dataset, sequence length, max-steps, text field, and CPU/SSD
  offload. Single-device start, or N-rank distributed (RDMA ring) — distributed
  requires CPU/SSD offload enabled.
- **Hierarchos (RWKV)** (`hierarchos-vulkan-train`): tokenizer, training data,
  ponder-loss weight, fp16. Single-device (via `hierarchos-native-cli`) or
  N-rank distributed (param-server / allreduce / zero1).

The **Distributed Training Launch** panel lets you tick which registered
GPUs/ranks to include; the rest of the config is reused from the shared fields
(epochs, batch, learning rate, seed, offload).

## Requirements for multi-node launch

- **SSH key auth** to each remote node (the dashboard runs `ssh <remote> ...`).
- The training binaries present on each node at the expected paths
  (`~/Hierarchos-Native/hierarchos-vulkan/target/release/...`).
- A working RDMA device per node (`rocep*` / `mlx5_*`, see `ibv_devinfo`).
- The RoCE subnets reachable between nodes (for the ring transport).

The dashboard only *launches and monitors*; it does not proxy GPU work.

## Log locations

- Dashboard server log: `dash.log` (next to `server.py`).
- Per-node training logs: `~/hierarchos-logs/<node_key>.log` (streamed into the
  browser log panels).
