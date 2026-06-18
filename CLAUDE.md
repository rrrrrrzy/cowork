# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

`cowork` is a coordination tool for working across multiple intranet development
machines that share the same NFS filesystem. Its two goals:

1. **Machine registry** — record the IP and hardware config of each intranet dev
   machine, so any node knows where the others are and what they can run.
2. **Experiment scheduling** — from one machine (a scheduler/head node), dispatch
   large-scale experiments across the registered machines and run them
   efficiently (GPU-aware placement, parallel fan-out, status tracking).

## Tooling in this repo

- **`machines.yaml`** — the machine registry (one entry per node: name, role,
  cpus, mem, gpus, conda roots/envs). Lives on `/share` so any node can read it
  without SSH. Seeded with `head` + two A800 workers. **Private connection
  details — real hostname, ssh host/port/user, internal_ip — are NOT in this
  file**; they live in `machines.local.yaml` (gitignored) and are merged onto
  each entry by node `name` at load time (`cw/registry.py`). Copy
  `machines.local.example.yaml` → `machines.local.yaml` to wire up a cluster.
- **`gpus.py`** — cluster GPU monitor (the substrate for GPU-aware placement).
  Reads `machines.yaml`, concurrently probes every node (local exec for our own
  node, `ssh` for the rest), runs `nvidia-smi`, and prints per-node / per-GPU
  usage **plus the tasks running on each GPU** (pid / user / GPU-mem / runtime /
  full command line — like `nvitop`, but across all machines). Standard-library
  only (no PyYAML / no `rich`); has its own minimal YAML-subset parser.

  ```bash
  ./gpus.py                  # DEFAULT: live GPU table (mem/util/free), refresh 1s
  ./gpus.py --procs          # also list per-GPU tasks (nvitop-style): pid/user/mem/runtime/cmd
  ./gpus.py --once           # render once and exit
  ./gpus.py --watch 5        # refresh every 5s instead of 1
  ./gpus.py --json           # machine-readable, one-shot; proc_list[] carries each task's cmd
  ./gpus.py --dump           # print the parsed registry and exit
  ./gpus.py --mem-thresh N   # MiB; a GPU with <= N used and no procs counts as "free"
  ```

  Remote probes reuse one persistent SSH connection per host (OpenSSH
  `ControlMaster`/`ControlPersist`, sockets in `/tmp/cowork-ssh-<uid>/`), so the
  default 1s `--watch` doesn't pay a fresh handshake every cycle; masters are
  torn down on exit. Disable with `--no-reuse` if a stale socket misbehaves.

  How the task view works: `nvidia-smi --query-compute-apps` only reports a
  truncated process name ("python"), so the probe joins GPU pids against
  `ps -o user,etime,args` on each node to recover the real command line.

  **Container caveat (confirmed 2026-06-08):** `head` is a bare host, so its
  tasks resolve fully (user / runtime / command). The two A800 workers, however,
  are reached by SSHing **into a PID-isolated pod** (`/proc` shows ~20 procs, no
  `hostPID`). The GPUs are shared at the node level, but the jobs using them run
  in **sibling containers** — so their pids don't exist in our namespace, `ps`
  finds nothing, and even `nvidia-smi` returns `[Not Found]` for the process
  name. For such tasks we can only show **pid + GPU-mem + util**; the command is
  unavailable and the view prints a note saying so. Recovering it would need
  node/host-level access (host SSH with `hostPID`, or `crictl`/`kubectl` on the
  node) — none of which is available from inside the pod (no runtime tools; the
  pod's k8s serviceaccount token is off-limits and wouldn't map pid→command
  anyway). For a single-machine deep dive on a bare host, `nvitop` is at
  `/share/ziyangrao/miniconda3/bin/nvitop`.

- **`cowork`** — the job scheduler: submit-and-forget task dispatch across the
  cluster. The head node never SSHes or checks GPUs; it just queues a job and the
  per-node agents do placement, conda activation, GPU pinning and keepbusy
  management. Standard-library only; logic lives in the `cw/` package, shared with
  `gpus.py` (one registry parser + one nvidia-smi probe).

  **→ Full usage reference (all commands, flags, state layout, recipes,
  troubleshooting): [`USAGE.md`](USAGE.md). Read it before operating cowork.**

  Architecture (file-based coordination on `/share`, no broker, no head daemon):
  - A **per-node agent** (`cowork agent`, started once via `cowork up`, lives in a
    tmux session) is the single authority for its node's GPUs. It watches the
    shared queue, **claims** any job it can fit via an atomic rename
    (`queued/<id>.json` → `running/<id>.json` — on NFSv4 rename is atomic, so
    exactly one agent wins, no locks), pins `CUDA_VISIBLE_DEVICES`, activates the
    job's conda env, and launches it **detached** (`setsid`) so the job survives an
    agent restart/redeploy. The job's bash wrapper writes its exit code to
    `state/logs/<id>.rc`, so completion is recoverable even across an agent restart
    (a re-adopted job is no longer our child).
  - The agent **runs keepbusy on its currently-free GPUs and SIGTERMs it the
    instant a real job needs them** — folding keepbusy into cowork instead of
    letting it guess from nvidia-smi. keepbusy is tracked by pid-file so a
    restarted agent reaps an orphan (keepbusy disguises its process name).
  - GPUs used by *external* (non-cowork) jobs are detected via a local nvidia-smi
    probe and excluded from placement, so cowork coexists with hand-run jobs.
  - State lives under `state/` on `/share` (job board + per-node heartbeats +
    logs); `cowork status` reads heartbeats, so it needs **no SSH**.

  ```bash
  ./cowork up                       # start agents on every node + watchdog (idempotent; tmux)
  ./cowork up --node head           # just one node (skips watchdog); --restart to force
  ./cowork submit --env libero --gpus 2 --cwd /share/ziyangrao/LIBERO -- python train.py
  ./cowork submit --gpu-ids 0,1 --name foo -- python x.py        # pin specific GPUs
  ./cowork submit --node node-8hlhv --gpu-ids 4,5 --env d3il -- python y.py  # node + GPUs
  ./cowork ps                       # queued + running   (--all adds done/failed)
  ./cowork logs <id> [-f] [--err]   # stream a job's stdout/stderr
  ./cowork cancel <id>              # queued -> drop; running -> SIGTERM its group
  ./cowork status                   # cluster free-GPU view from heartbeats (no SSH)
  ./cowork down                     # stop agents + watchdog (detached jobs keep running)
  ./cowork watch [--interval SEC]   # supervisor loop (auto-started by `up`)
  ./cowork agent [--node N]         # run the daemon in foreground (used by `up`)
  ```

  Pinning: `--node NAME` pins a job to one node; `--gpu-ids 0,1` pins exact
  physical GPUs; combine them for "this node, these GPUs". Out-of-range pins are
  rejected at submit time (so a job can't queue forever asking for GPUs a node
  doesn't have).

  **Idle-shutdown safety net.** This cluster reclaims a node after **25 minutes**
  of GPU idleness (verified 2026-06-18). cowork stays well under that: the only
  uncovered gap is "the agent itself is down", so `cowork up` also starts a
  **watchdog** (`cowork watch`, a tmux loop on the head) that re-ups any agent
  whose heartbeat goes stale — default every 300s, ~5.5 min worst-case idle, a
  ~4.5x margin. keepbusy's own warm-up gap (~30s after a GPU frees) is negligible
  against 25 min, so `idle_minutes` is left at 0.5 (debounce) rather than 0. There
  is no cron in these pods, so the watchdog is a tmux loop, not a cron job; a node
  reboot needs a manual `cowork up` (same as the standalone keepbusy today).

  Config is an optional `cowork:` block in `machines.yaml` (every field has a
  default): `state_dir`, `heartbeat_interval`, `poll_interval`, `mem_thresh`, and
  `keepbusy: {enabled, env, run_py, idle_minutes}`. Env overrides: `COWORK_STATE`
  (board location — `up` forwards it to remote agents explicitly, since tmux does
  not propagate arbitrary env vars), `COWORK_KEEPBUSY=0` (disable keepbusy).

  No build step (stdlib only). Quick test on the head node without touching real
  GPUs: `COWORK_STATE=/tmp/cw COWORK_KEEPBUSY=0 ./cowork agent --node head &` then
  `COWORK_STATE=/tmp/cw ./cowork submit --gpus 0 -- bash -c 'echo hi'` and watch
  `cowork ps --all`. Verified end-to-end 2026-06-18: claim → run → reap with exit
  codes, GPU pinning, cancel, agent-restart re-adoption of detached jobs, and
  keepbusy preempt/restore.

## Why the shared filesystem is central

`/share` is a **shared NFS4 mount** (a managed CFS export), mounted at
the same path on every machine in this cluster. Anything written under
`/share/ziyangrao/cowork` is **immediately visible on all nodes**. This makes the
shared filesystem the simplest coordination substrate:

- The machine registry can be a plain file here (no service to host).
- A job queue / status board can be files + atomic renames here, readable by every
  node — no message broker required.
- Code and configs don't need to be `rsync`'d to each node; they're already shared.

Design the scheduler to lean on this: prefer file-based coordination on `/share`
over standing up a network service, unless you specifically need low latency.

## Cluster environment (discovered, verify before relying on it)

This head machine, as inspected on init (real hostname/IP live in
`machines.local.yaml`, not here):

- On the intranet, reachable directly on port 22; workers go through a gateway
  port-forward (see `machines.local.yaml`).
- 180 vCPU, ~1 TiB RAM, **4× NVIDIA B20Z** GPU.
- conda available at `/share/ziyangrao/miniconda3` and `/share/ziyangrao/miniforge3`;
  named envs (shared via NFS): `d3il`, `fmmi`, `libero`, `mmsae`, `vla_engine_developer`.

Re-discover these on any node with: `hostname -I`, `nproc`, `free -h`, `nvidia-smi -L`.
Do not hardcode this one machine's values into the registry — capture them per node.

### Available vs. missing tooling

- Present: `ssh`, `tmux`, `git`, `conda`, `nvidia-smi`.
- **Missing: `pdsh`, `rsync`.** Don't assume a parallel-ssh or rsync workflow.
  Fan out with `ssh` loops (or a small Python `subprocess`/`asyncio` driver), and
  rely on the shared NFS instead of `rsync` for moving files between nodes.

## Workloads being scheduled

The experiments this tool dispatches live in sibling directories under
`/share/ziyangrao/` and are mostly Vision-Language-Action / robotics research:
`AlphaBrain`, `LIBERO`, `lerobot`, `openpi`, `vla-interp`, `d3il`, `mm_sae`,
`nmi_tcbm`. They are GPU-bound and run inside the conda envs above — so the
scheduler must be GPU-aware (track free GPUs per node, pin `CUDA_VISIBLE_DEVICES`)
and env-aware (activate the right conda env before launching a job).

Related existing helpers (not part of this repo, but relevant):

- `/share/ziyangrao/keepbusy/` — GPU idle monitor that runs synthetic load to keep
  GPUs warm, and **yields when real compute appears**. A scheduler that places jobs
  must coexist with this (real jobs should preempt keepbusy automatically).
- `/share/ziyangrao/server_clash/` — clash proxy setup for outbound internet from
  the server; source its `proxy_functions.sh` if a job needs external network.

## Conventions to establish (no code enforces these yet)

When you build out the tool, keep it discoverable for future sessions:

- Store the machine registry as a single human-readable file (e.g. `machines.yaml`
  or `machines.json`) in this directory, one entry per node with at least:
  `name`, `ip`, `cpus`, `mem`, `gpus` (model + count), `conda_root`. Seed it with
  this head node as the first entry.
- Keep per-experiment job state under a predictable path on `/share` so any node
  can read scheduler status without SSH.
- Make the dispatch path GPU-aware and conda-env-aware from the start; retrofitting
  GPU placement later is painful.
