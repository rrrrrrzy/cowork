# cowork

Coordination tooling for a small cluster of intranet dev machines that share one
NFS filesystem. It does two things:

1. **Machine registry** — record each node's hardware and conda setup in one
   shared file, so every node knows what the others can run.
2. **Experiment scheduling** — submit GPU-bound jobs from any node and let
   per-node agents place them: GPU-aware, conda-env-aware, fire-and-forget.

It's built for VLA / robotics research workloads (LIBERO, openpi, lerobot, …)
that run inside conda envs on shared GPUs, where the only common substrate is the
NFS mount — no message broker, no head daemon, no `rsync`.

## Why file-based

`/share` is mounted at the same path on every node, so anything written under it
is instantly visible everywhere. cowork leans on this entirely:

- the registry is a plain YAML file,
- the job queue is a directory of JSON files,
- jobs are **claimed by atomic rename** (`queued/<id>.json` → `running/<id>.json`;
  NFSv4 rename is atomic, so exactly one agent wins — no locks, no broker),
- status is read from per-node heartbeat files (so `cowork status` needs no SSH).

## Components

| File | What it is |
|---|---|
| `machines.yaml` | The machine registry — one entry per node (cpus, mem, gpus, conda). Non-sensitive only; private connection details (ssh host/port/user, internal IPs) live in a gitignored `machines.local.yaml` overlay, merged by node name at load time. |
| `gpus.py` | Cluster GPU monitor (read-only). Probes every node concurrently, runs `nvidia-smi`, and prints a live per-GPU table — optionally with the task on each GPU (pid/user/mem/runtime/cmd), like `nvitop` across all machines. |
| `cowork` | The job scheduler CLI. Submit-and-forget dispatch; all logic lives in the `cw/` package, shared with `gpus.py` (one registry parser + one nvidia-smi probe). |
| `cw/` | The scheduler internals: `registry`, `probe`, `state`, `agent`, `cli`. |
| `keepbusy/` | Vendored GPU idle-filler. The agent runs it on free GPUs and SIGTERMs it the instant a real job needs them. |

Standard-library Python only — no build step, no third-party deps.

## Architecture

A **per-node agent** (started once via `cowork up`, lives in a tmux session) is
the single authority for its node's GPUs. It watches the shared queue, claims any
job it can fit, pins `CUDA_VISIBLE_DEVICES`, activates the job's conda env, and
launches it **detached** (`setsid`) so the job survives an agent restart. The
job's exit code is written to disk, so completion is recoverable even if the agent
is bounced. The head node never SSHes or polls GPUs — it just queues files.

A **watchdog** (`cowork watch`, also auto-started by `cowork up`) re-ups any agent
whose heartbeat goes stale, keeping the cluster well under its idle-reclaim window.
GPUs used by *external* (non-cowork) jobs are detected via a local `nvidia-smi`
probe and excluded from placement, so cowork coexists with hand-run jobs.

## Quickstart

```bash
# 1. wire up your cluster (once)
cp machines.local.example.yaml machines.local.yaml   # fill in ssh host/port/user per node
# edit machines.yaml to match your nodes' hardware/conda

# 2. start the agents + watchdog (idempotent; runs in tmux)
./cowork up

# 3. submit jobs from any node — returns immediately with a job id
./cowork submit --env libero --gpus 2 --cwd /share/ziyangrao/LIBERO -- python train.py
./cowork submit --node node-8hlhv --gpu-ids 4,5 --env d3il -- python y.py

# 4. observe
./cowork ps                 # queued + running   (--all adds done/failed)
./cowork logs <id> -f       # follow a job's output
./cowork status             # cluster free-GPU view, from heartbeats (no SSH)
./cowork cancel <id>        # drop a queued job, or SIGTERM a running one
./cowork down               # stop agents + watchdog (detached jobs keep running)
```

GPU monitor (separate from the scheduler):

```bash
./gpus.py            # live per-GPU table (mem/util/free), refresh 1s
./gpus.py --procs    # also list per-GPU tasks (nvitop-style)
./gpus.py --once     # render once and exit
./gpus.py --json     # machine-readable, one-shot
```

## Documentation

- **[`USAGE.md`](USAGE.md)** — full command reference (every flag, state layout,
  recipes, troubleshooting). Read this before operating cowork.
- **[`CLAUDE.md`](CLAUDE.md)** — design rationale, cluster environment notes, and
  the confirmed caveats (e.g. PID-isolated worker pods, idle-reclaim timing).
- **[`keepbusy/README.md`](keepbusy/README.md)** — the vendored idle-filler.

## Configuration

The registry's optional `cowork:` block tunes the scheduler (every field has a
default): `state_dir`, `heartbeat_interval`, `poll_interval`, `mem_thresh`, and a
`keepbusy: {enabled, env, run_py, idle_minutes}` sub-block. Env overrides:
`COWORK_STATE` (board location) and `COWORK_KEEPBUSY=0` (disable keepbusy).
