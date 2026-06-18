# keepbusy (vendored)

A GPU idle monitor: it watches `nvidia-smi`, and **only after a GPU has stayed
idle for a sustained window** it runs a synthetic PyTorch workload to keep the
GPU warm — then **yields the instant real compute appears**. cowork uses it to
keep nodes above the cluster's idle-shutdown threshold while real jobs always
take precedence.

This is a **vendored copy** of the standalone `keepbusy` tool, copied into the
cowork repo so the scheduler is self-contained: the per-node agent depends on
`run.py` at launch time, and pinning the exact code here (versioned with the
agent that drives it) removes a silent external dependency. The cowork agent
launches it from this directory — see [`../USAGE.md`](../USAGE.md) and
[`../CLAUDE.md`](../CLAUDE.md).

> Upstream lives at `/share/ziyangrao/keepbusy/`. If you change behaviour here,
> keep the two in sync (or retire the upstream copy).

## Files

- **`keepbusy.py`** — the monitor + synthetic workload. Requires `torch` with
  CUDA. Polls `utilization.gpu`; once every selected GPU has been below
  `--util-threshold` for `--idle-minutes`, it trains a throwaway MLP
  (`ComplexModel`, `DataParallel` across the visible GPUs) in a tight loop.
- **`run.py`** — a thin launcher that runs `keepbusy.py` under a **disguised
  process name** (default `python3`, override with `KB_PROC_NAME`). It re-execs
  itself with a rewritten `argv[0]` and applies `prctl(PR_SET_NAME)` +
  `setproctitle` + an `/proc/self/cmdline` overwrite, so the load is
  indistinguishable from an ordinary interpreter in `ps`/`top`/`nvidia-smi`.
  `run.py` does `import keepbusy`, so **both files must stay in the same
  directory.**

## How cowork drives it

The agent (`cw/agent.py`, `_set_keepbusy`) runs **one keepbusy process per node**
over that node's currently-free GPUs, and SIGTERMs it the moment a real job needs
any of them. It is invoked roughly as:

```bash
<conda base python> keepbusy/run.py \
    --gpus 0,1,2,3,4,5,6,7 \   # the node's free physical GPU ids
    --idle-minutes 0.5 \       # cowork:keepbusy.idle_minutes in machines.yaml
    --poll-interval 1
```

- The Python interpreter comes from the `cowork.keepbusy.env` conda env (default
  `base`, which has `torch`); the path to `run.py` is `cowork.keepbusy.run_py`
  in `machines.yaml` (now points at this in-repo copy).
- keepbusy is tracked by a **pid-file** (`state/agents/<node>.keepbusy.pid`) so a
  restarted agent can reap an orphan even though the process name is disguised.
- keepbusy's own **auto-yield** (it returns to the monitor stage when it sees ≥2
  GPU compute PIDs, releasing cached CUDA memory) is a second line of defense:
  even if the agent were slow to preempt, keepbusy steps aside for a real job.

## ⚠️ Caveat in PID-isolated pods (read before touching GPU/proc detection)

On the bare-host head node, keepbusy's GPU process shows up in
`nvidia-smi --query-compute-apps` under the **same pid** the agent launched, and
`ps` resolves it. Inside the A800 **worker pods** (PID-isolated, no `hostPID`),
the driver reports keepbusy's GPU process under its **host-namespace pid**, which
does **not** match the pod-namespace pid the agent holds, and `ps` in the pod
can't resolve it (`resolved=False`, name `[Not Found]`).

Consequence: **you cannot identify "our own keepbusy" by pid inside a pod.** An
earlier version of the agent did exactly that to spot external jobs on keepbusy
GPUs — and mis-flagged keepbusy's own load as an external job every cycle,
thrashing keepbusy in a restart loop so it never sustained load. The GPUs then
sat at 0% util and the node was reclaimed by the idle-shutdown. The agent now
only treats a **resolvable** foreign process as external on a keepbusy GPU (see
`_compute_free` in `cw/agent.py`). Keep that in mind before reworking GPU/process
detection.

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--gpus` | (all visible) | Comma-separated physical GPU indices, e.g. `0,1`. Restricts both monitoring and the workload (sets `CUDA_VISIBLE_DEVICES`). |
| `--idle-minutes` | 10.0 | Required continuous idle time before keepbusy starts. cowork sets `0.5`. |
| `--util-threshold` | 5.0 | A GPU counts as idle below this utilization %. |
| `--poll-interval` | 5.0 | Seconds between utilization checks. cowork sets `1`. |
| `--scale` | 2000 | Model size factor for the synthetic workload. |
| `--batch-size` | 6400 | Synthetic batch size. |
| `--disable-auto-interrupt` | off | Keep running even if other GPU compute processes appear (disables auto-yield). |

## Requirements

- Python 3.8+, **PyTorch with CUDA** (present in the `base` conda env on `/share`).
- NVIDIA drivers with `nvidia-smi` on `PATH`.

## Standalone usage

```bash
# disguised (what cowork uses)
python run.py --gpus 0,1 --idle-minutes 0.5 --poll-interval 1
KB_PROC_NAME=worker python run.py --gpus 0,1

# undisguised
python keepbusy.py --gpus 0,1 --idle-minutes 5
```

Stop with `Ctrl+C` / SIGTERM — it releases CUDA memory before exiting.
