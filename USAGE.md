# cowork — usage reference

`cowork` is submit-and-forget job scheduling across the shared-NFS dev cluster.
**You (on any node, usually `head`) only queue jobs.** Per-node agents do GPU-aware
placement, conda activation, `CUDA_VISIBLE_DEVICES` pinning, keepbusy management,
and status reporting. No SSH or GPU-checking from the submitter.

Everything coordinates through files under `state/` on `/share` (visible on every
node). Standard-library Python only; the entrypoint is `./cowork` (runs under the
shared miniconda python, identical on every node).

- Quick orientation + design rationale: `CLAUDE.md`.
- GPU monitor (read-only, not the scheduler): `./gpus.py` (see its `--help`).

---

## TL;DR

```bash
./cowork up                                   # one-time: start agents on all nodes + watchdog
./cowork submit --env libero --gpus 1 --cwd /share/ziyangrao/LIBERO -- python train.py
./cowork ps                                   # what's queued / running
./cowork logs <id> -f                         # follow a job's output
./cowork status                               # cluster free-GPU view (no SSH)
```

`submit` prints a job id and returns immediately. An agent claims it within a few
seconds, pins GPUs, activates the env, and runs it.

---

## Commands

Global form: `./cowork [--registry PATH] <command> [options]`
(`--registry` defaults to `./machines.yaml`.)

### `submit … -- CMD…` — queue a job

```bash
./cowork submit [options] -- COMMAND ARGS...
```

| Option | Default | Meaning |
|---|---|---|
| `--env ENV` | `base` | conda env to activate before running |
| `--gpus N` | `1` | number of GPUs to request (`0` = CPU-only job) |
| `--gpu-ids 0,1` | — | pin to **specific physical GPU ids** (overrides `--gpus`) |
| `--node NAME` | any | pin to a specific node (e.g. `node-8hlhv`) |
| `--name NAME` | cmd[0] | human label shown in `ps`/`status` |
| `--cwd DIR` | current dir | working directory for the job |
| `--conda-root R` | registry's first | conda root to source |
| `--env-var K=V` | — | extra env var for the job (repeatable) |
| `--priority P` | `0` | reserved (ordering is currently FIFO by id) |
| `-q, --quiet` | off | print only the job id (for scripting) |

Everything after `--` is the command, run verbatim (its own flags are not parsed
by cowork). The job runs as:
`conda activate ENV && cd CWD && CUDA_VISIBLE_DEVICES=… <CMD>`.

Pinning:
- `--node X` → only node X may run it.
- `--gpu-ids 0,1` → exactly physical GPUs 0 and 1 (on whichever node, or on X if
  combined with `--node X`).
- `--node X --gpu-ids 4,5` → "this node, these GPUs".
- Out-of-range pins are rejected **at submit time** (a job can't queue forever
  asking for GPUs a node doesn't have).

Examples:
```bash
./cowork submit --env d3il --gpus 2 --name d3il-sweep --cwd /share/ziyangrao/d3il -- \
    python train.py --config base
./cowork submit --node node-zt2bc --gpu-ids 0,1,2,3 --env libero -- bash run.sh
./cowork submit --gpus 0 --name preprocess -- python prep.py        # CPU-only
./cowork submit --env mmsae --env-var WANDB_MODE=offline -- python x.py
```

### `ps` — list jobs

```bash
./cowork ps          # queued + running
./cowork ps --all    # also done + failed
```
Columns: `ID  STATE  NODE  GPUS  ENV  AGE  NAME`. `GPUS` shows assigned ids once
running, or the request (`x2`, or `0,1?`) while queued.

### `logs <id>` — show a job's output

```bash
./cowork logs <id>          # stdout (full)
./cowork logs <id> -f       # follow (stops when the job finishes)
./cowork logs <id> --err    # stderr instead of stdout
```
`<id>` may be a unique prefix. Logs live at `state/logs/<id>.{out,err}`.

### `cancel <id>` — cancel a job

```bash
./cowork cancel <id>
```
Queued → dropped to `failed`. Running → the owning agent SIGTERMs the job's whole
process group (then SIGKILL if needed) and marks it `failed (cancelled)`.

### `status` — cluster view (no SSH)

```bash
./cowork status
```
Reads agent heartbeats from `state/agents/`. Shows per-node `up`/`DOWN`/`stopped`,
free GPU ids, running jobs, keepbusy GPUs, and the cluster free-GPU total. Job
counts (queued/running/done/failed) at the top. Instant — never touches SSH.

### `up` / `down` — bring agents up / take them down

```bash
./cowork up                       # start an agent on every node (tmux) + the watchdog
./cowork up --node head           # only this node (does NOT start the watchdog)
./cowork up --restart             # force-restart agents even if already up
./cowork up --no-watch            # start agents but not the watchdog
./cowork up --watch-interval 120  # watchdog checks every 120s (default 300)

./cowork down                     # stop watchdog + all agents
./cowork down --node node-8hlhv   # stop just one agent (watchdog will revive it)
```
- Idempotent: a healthy agent is left alone.
- **Running jobs are detached (`setsid`)** — `down`, agent restarts, and redeploys
  do **not** kill them. A restarted agent re-adopts its still-running jobs and
  reaps any that finished while it was away (exit code recovered from
  `state/logs/<id>.rc`).
- A full `down` stops the watchdog first (so it can't immediately re-up what you
  just stopped). A single-node `down` leaves the watchdog running.

### `watch` — the supervisor loop (auto-started by `up`)

```bash
./cowork watch                 # loop: re-up any agent whose heartbeat goes stale
./cowork watch --interval 60   # check every 60s (default 300)
./cowork watch --once          # single check, then exit
```
`up` starts this in its own tmux session (`cowork-watch`) on the head. It is the
safety net against the cluster's **25-minute idle-shutdown**: the only uncovered
idle gap is "the agent itself is down", and the watchdog closes it within one
interval. Default 300s ⇒ ~5.5 min worst-case idle, ~4.5× margin.

### `agent` — the per-node daemon (used by `up`)

```bash
./cowork agent [--node NAME]   # foreground; node auto-detected by hostname if omitted
```
You normally never run this directly — `up` launches it in tmux. Run it in the
foreground only to debug an agent.

---

## State layout (`state/`, on `/share`)

```
state/
  jobs/
    queued/<id>.json     # submitted, awaiting an agent
    running/<id>.json    # claimed + launched (has pid/pgid/gpu_ids/node)
    running/<id>.cancel  # cancel request marker (any node can drop this)
    done/<id>.json       # exit 0
    failed/<id>.json     # nonzero exit, crash, or cancelled
  agents/
    <node>.json          # heartbeat: alive_ts, free GPUs, running jobs, keepbusy
    <node>.keepbusy.pid  # tracks keepbusy so a restarted agent can reap an orphan
  logs/
    <id>.out / <id>.err  # job stdout / stderr
    <id>.rc              # job exit code (written by the job's wrapper)
    agent-<node>.log     # an agent's own diagnostics
    keepbusy-<node>.log  # keepbusy output
    watch.log            # the watchdog's log
```

A job moves between states by **atomic `os.rename`** (NFSv4 rename is atomic, so a
claim has exactly one winner — no locks). To inspect a job directly, read its
`.json`; to see why one failed, read `state/logs/<id>.err` and the `reason` field.

Job-state lifecycle: `queued → running → done | failed`. Cancel: queued→failed, or
running→(SIGTERM)→failed.

---

## Configuration

Optional `cowork:` block in `machines.yaml` (every field has a default):

```yaml
cowork:
  state_dir: state           # board location, relative to repo (on /share)
  heartbeat_interval: 5       # staleness reference for "agent DOWN"
  poll_interval: 3            # agent claim/reap cycle period (s)
  mem_thresh: 2048            # MiB; a GPU at/under this with no procs counts as free
  heartbeat_stale_factor: 4   # heartbeat older than factor*interval => DOWN
  keepbusy:
    enabled: true
    env: base                 # a conda env that has torch
    run_py: /share/ziyangrao/keepbusy/run.py
    idle_minutes: 0.5         # idle time before keepbusy fills a freed GPU
```

Environment overrides:
- `COWORK_STATE=/path` — use a different board (e.g. for testing). `up` forwards
  this to remote agents explicitly (tmux does not propagate arbitrary env vars).
- `COWORK_KEEPBUSY=0` (or `off`/`false`/`no`) — disable keepbusy supervision.

---

## How placement & keepbusy work (operator's mental model)

- Each agent owns its node's GPUs. Every cycle it computes **free GPUs** = total −
  GPUs used by cowork jobs − GPUs an **external** (non-cowork) process is using
  (detected via local `nvidia-smi`). So cowork coexists with hand-run jobs: if you
  launch a GPU process **not** through cowork, the agent sees it (it identifies its
  own keepbusy by pid, so anything else is "external"), drops that GPU from the
  pool, **re-scopes keepbusy off it**, and never schedules a cowork job there. When
  your process exits, the GPU returns to the pool and keepbusy resumes — all within
  one poll cycle (`poll_interval`, default 3s).
- It claims a queued job it can fit, pins the GPUs, and launches it detached.
- It runs **keepbusy on the free GPUs** and **SIGTERMs it the instant a real job
  needs them** (keepbusy frees CUDA cache on SIGTERM), then restarts keepbusy on
  whatever's still free when the job ends. keepbusy GPUs are always "free to
  claim" — keepbusy is preemptible.
- Heartbeats published every cycle drive `status` and the watchdog.

---

## Recipes

**First-time bring-up (and after a node reboot):**
```bash
./cowork up            # starts agents on all nodes + the watchdog
./cowork status        # confirm every node is "up"
```

**Run an experiment and watch it:**
```bash
id=$(./cowork submit --env libero --gpus 2 --cwd /share/ziyangrao/LIBERO -q -- python train.py)
./cowork logs $id -f
```

**Fan out many jobs (they spread across free GPUs cluster-wide automatically):**
```bash
for cfg in a b c d; do
  ./cowork submit --env d3il --gpus 1 --name "sweep-$cfg" --cwd /share/ziyangrao/d3il -q -- \
    python train.py --config $cfg
done
./cowork ps
```

**Pin to a specific node and GPUs:**
```bash
./cowork submit --node node-zt2bc --gpu-ids 0,1,2,3 --env libero -- bash run.sh
```

**Tear everything down (jobs keep running):**
```bash
./cowork down
```

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| Job stuck `queued` | No agent up (`./cowork status`, then `./cowork up`); or it asked for more GPUs than are free anywhere; or a `--node` pin has no free GPUs right now. |
| `submit` rejected | `--gpu-ids`/`--gpus` exceed the target node's GPU count — fix the request. |
| Node shows `DOWN` | Agent died; `./cowork up` (the watchdog also re-ups it within its interval). |
| Node reboot | No cron in these pods — run `./cowork up` manually after a reboot. |
| Job `failed` unexpectedly | Read `state/logs/<id>.err` and the `reason` in `state/jobs/failed/<id>.json`. |
| Agent misbehaving | Read `state/logs/agent-<node>.log`; `./cowork up --node X --restart`. |
| Watchdog issues | Read `state/logs/watch.log`. |
| Don't want keepbusy | `COWORK_KEEPBUSY=0 ./cowork up` (or set `keepbusy.enabled: false`). |
| Stale SSH socket | cowork reuses OpenSSH ControlMaster sockets in `/tmp/cowork-ssh-<uid>/`; delete them if one wedges. |

Worker container caveat: the A800 workers are reached via a PID-isolated pod, so
GPU **memory/util** is always visible but a sibling-container job's **command line**
is not (see `CLAUDE.md`). cowork's own jobs resolve fully; external jobs on a
worker are still correctly counted as "busy" for placement.

---

## Notes for agents editing this repo

- `machines.yaml` is parsed by a **block-style-only** mini-YAML parser
  (`cw/registry.py`) — no flow style (`{a: 1}`), no multi-doc. Booleans
  `true/false`, `null`, ints, floats, and `&anchors`/`*aliases` (aliases left
  unresolved) are supported.
- The package is `cw/` (`registry`, `probe`, `state`, `agent`, `cli`); `gpus.py`
  and `cowork` both import from it. Keep stdlib-only (runs on every node's shared
  miniconda python; no pip installs assumed).
- Jobs are **detached**; never assume killing an agent stops its jobs. Job exit
  status is the `.rc` sentinel file, not a waited child (a re-adopted job isn't
  our child).
- Default `--cwd` is the submitter's current dir — for reproducibility, pass an
  absolute `--cwd`.
