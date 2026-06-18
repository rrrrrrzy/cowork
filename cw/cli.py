"""The `cowork` command surface.

Submit-and-forget UX for the head node: write a job file to the shared queue and
return. Placement, GPU pinning, conda activation and keepbusy are all handled by
the per-node agents (see cw/agent.py) — the submitter never SSHes or checks GPUs.

    cowork submit [opts] -- CMD...   queue a job, print its id
    cowork ps [--all]                list jobs (queued/running, +done/failed)
    cowork logs <id> [-f] [--err]    stream a job's output
    cowork cancel <id>               cancel a queued or running job
    cowork status                    cluster view from agent heartbeats (no SSH)
    cowork up [--node N] [--restart] start agents on the nodes (idempotent)
    cowork down [--node N]           stop agents (running jobs keep running)
    cowork agent [--node N]          run the per-node daemon (used by `up`)
"""
import argparse
import os
import shlex
import sys
import time
from pathlib import Path

from . import agent as agent_mod
from . import probe, state
from .registry import load_registry, machine_by_name, machines

ENTRY = str(state.REPO / "cowork")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fmt_age(ts):
    if not ts:
        return "-"
    s = max(0, int(time.time() - ts))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    if s < 86400:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 86400}d{(s % 86400) // 3600:02d}h"


def _fmt_gib(mib):
    """MiB int -> compact GiB string, e.g. 8200 -> '8G', 81920 -> '80G'."""
    if not mib:
        return "0G"
    g = mib / 1024
    return f"{g:.0f}G" if g >= 10 else f"{g:.1f}G"


def _util_bar(util, width=10):
    """Tiny colored utilization bar: green when warm/busy, dim when idle."""
    if not isinstance(util, int):
        return "[" + "?" * width + "]"
    fill = max(0, min(width, round(util / 100 * width)))
    color = "\033[32m" if util >= 50 else ("\033[33m" if util >= 5 else "\033[90m")
    return f"{color}[{'#' * fill}{'.' * (width - fill)}]\033[0m"


def _gpus_str(spec):
    if spec.get("assigned_gpu_ids") is not None:
        return ",".join(map(str, spec["assigned_gpu_ids"])) or "-"
    if spec.get("gpu_ids"):
        return ",".join(map(str, spec["gpu_ids"])) + "?"
    return f"x{spec.get('gpus', 1)}"


def _resolve(cfg, ref, states=state.JOB_STATES):
    hit = state.find_job(cfg, ref, states)
    if hit is None:
        sys.exit(f"no job matching {ref!r}")
    if hit == "ambiguous":
        sys.exit(f"{ref!r} matches multiple jobs; use a longer prefix")
    return hit


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def cmd_submit(args, reg, cfg):
    if not args.cmd:
        sys.exit("nothing to run: put the command after `--`, e.g. "
                 "cowork submit --env libero -- python train.py")
    if args.gpu_ids:
        try:
            gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
        except ValueError:
            sys.exit(f"--gpu-ids expects a comma list of ints, got {args.gpu_ids!r}")
        if len(set(gpu_ids)) != len(gpu_ids):
            sys.exit(f"--gpu-ids has duplicates: {args.gpu_ids!r}")
        gpus = len(gpu_ids)
    else:
        gpu_ids = None
        gpus = args.gpus
    env_vars = {}
    for kv in args.env_var or []:
        if "=" not in kv:
            sys.exit(f"--env-var expects KEY=VALUE, got {kv!r}")
        k, v = kv.split("=", 1)
        env_vars[k] = v
    # Resolve / validate the node pin so a job can't sit queued forever because
    # it asks for GPUs the target node doesn't have.
    machine = None
    if args.node:
        machine = machine_by_name(reg, args.node)
        if machine is None:
            sys.exit(f"--node {args.node!r} is not in the registry "
                     f"(known: {', '.join(m['name'] for m in machines(reg))})")
        count = int((machine.get("gpus") or {}).get("count", 0) or 0)
        if gpu_ids is not None and any(i < 0 or i >= count for i in gpu_ids):
            sys.exit(f"--gpu-ids {gpu_ids} out of range for node {args.node!r} "
                     f"(has GPUs 0..{count - 1})")
        if gpu_ids is None and gpus > count:
            sys.exit(f"--gpus {gpus} exceeds node {args.node!r} GPU count ({count})")
    elif gpu_ids is not None:
        # gpu-ids without --node: only nodes that physically have all those ids
        # can ever run it; warn if NO node could (otherwise it'd queue forever).
        maxid = max(gpu_ids)
        if all(maxid >= int((m.get("gpus") or {}).get("count", 0) or 0)
               for m in machines(reg)):
            sys.exit(f"--gpu-ids {gpu_ids}: no node has a GPU {maxid}; "
                     f"pin a node with --node or lower the ids")
    conda_root = args.conda_root or (reg.get("conda_roots") or [None])[0]
    cwd = str(Path(args.cwd).resolve()) if args.cwd else os.getcwd()

    jid = state.new_job_id()
    spec = {
        "id": jid,
        "name": args.name or (args.cmd[0] if args.cmd else jid),
        "cmd": list(args.cmd),
        "env": args.env,
        "conda_root": conda_root,
        "cwd": cwd,
        "gpus": gpus,
        "gpu_ids": gpu_ids,
        "node": args.node,
        "env_vars": env_vars,
        "priority": args.priority,
        "submitted_by": os.environ.get("USER", "?"),
        "submitted_at": state.now_iso(),
        "submitted_ts": time.time(),
    }
    state.ensure_dirs(cfg)
    state.atomic_write_json(state.job_path(cfg, "queued", jid), spec)
    print(jid)
    if not args.quiet:
        want = ",".join(map(str, gpu_ids)) if gpu_ids else f"{gpus} gpu(s)"
        print(f"queued: {spec['name']}  env={args.env}  want={want}"
              + (f"  node={args.node}" if args.node else ""), file=sys.stderr)


# --------------------------------------------------------------------------- #
# ps
# --------------------------------------------------------------------------- #
def cmd_ps(args, reg, cfg):
    states = state.JOB_STATES if args.all else ("queued", "running")
    rows = []
    for st, j in state.iter_jobs(cfg, states):
        if st == "running":
            age = _fmt_age(j.get("started_ts"))
        elif st in ("done", "failed"):
            age = _fmt_age(j.get("finished_ts"))
        else:
            age = _fmt_age(j.get("submitted_ts"))
        rows.append((
            j.get("id", "?"), st, j.get("assigned_node") or j.get("node") or "-",
            _gpus_str(j), j.get("env", "-"), age,
            (j.get("name") or "")[:30],
        ))
    if not rows:
        print("no jobs" + ("" if args.all else " (try --all)"))
        return
    rows.sort(key=lambda r: r[0])
    hdr = ("ID", "STATE", "NODE", "GPUS", "ENV", "AGE", "NAME")
    widths = [max(len(str(r[i])) for r in rows + [hdr]) for i in range(len(hdr))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*hdr))
    color = {"running": "\033[33m", "done": "\033[32m", "failed": "\033[31m",
             "queued": "\033[36m"}
    for r in rows:
        c = color.get(r[1], "")
        line = fmt.format(*r)
        if c and sys.stdout.isatty():
            line = line.replace(r[1], f"{c}{r[1]}\033[0m", 1)
        print(line)


# --------------------------------------------------------------------------- #
# logs
# --------------------------------------------------------------------------- #
def cmd_logs(args, reg, cfg):
    _st, j = _resolve(cfg, args.id)
    lp = state.log_paths(cfg, j["id"])
    path = lp["err"] if args.err else lp["out"]
    if not path.exists():
        print(f"(no {'stderr' if args.err else 'stdout'} yet for {j['id']})")
        return
    if not args.follow:
        sys.stdout.write(path.read_text())
        return
    with open(path) as f:
        sys.stdout.write(f.read())
        sys.stdout.flush()
        try:
            while True:
                line = f.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    hit = state.find_job(cfg, j["id"])
                    if hit and hit[0] in ("done", "failed"):
                        time.sleep(0.3)
                        sys.stdout.write(f.read())
                        break
                    time.sleep(0.4)
        except KeyboardInterrupt:
            pass


# --------------------------------------------------------------------------- #
# cancel
# --------------------------------------------------------------------------- #
def cmd_cancel(args, reg, cfg):
    st, j = _resolve(cfg, args.id, states=("queued", "running"))
    jid = j["id"]
    if st == "queued":
        rec = dict(j)
        rec.update({"finished_at": state.now_iso(), "finished_ts": time.time(),
                    "exit_code": None, "reason": "cancelled (was queued)"})
        # Move out of queued atomically so no agent can claim it mid-cancel.
        if state.claim(Path(j["_path"]), state.job_path(cfg, "failed", jid)):
            state.atomic_write_json(state.job_path(cfg, "failed", jid), rec)
            print(f"cancelled queued job {jid}")
        else:
            print(f"{jid} was just claimed by an agent; sending cancel marker")
            state.cancel_marker(cfg, jid).touch()
    else:
        state.cancel_marker(cfg, jid).touch()
        print(f"cancel requested for running job {jid} "
              f"(agent on {j.get('assigned_node')} will stop it)")


# --------------------------------------------------------------------------- #
# status
# --------------------------------------------------------------------------- #
def cmd_status(args, reg, cfg):
    agents = {a["node"]: a for a in state.read_agents(cfg)}
    counts = {s: 0 for s in state.JOB_STATES}
    for st, _j in state.iter_jobs(cfg, state.JOB_STATES):
        counts[st] += 1
    print(f"jobs: \033[36m{counts['queued']} queued\033[0m  "
          f"\033[33m{counts['running']} running\033[0m  "
          f"\033[32m{counts['done']} done\033[0m  "
          f"\033[31m{counts['failed']} failed\033[0m\n")
    total_free = total_gpu = 0
    for m in machines(reg):
        node = m.get("name")
        a = agents.get(node)
        gtot = int((m.get("gpus") or {}).get("count", 0) or 0)
        total_gpu += gtot
        if a is None:
            print(f"\033[1m{node:<12}\033[0m  \033[90mno agent (run `cowork up`)\033[0m")
            continue
        stale = state.agent_is_stale(cfg, a)
        if a.get("stopped"):
            tag = "\033[90mstopped\033[0m"
        elif stale:
            tag = f"\033[31mDOWN\033[0m (last seen {_fmt_age(a.get('alive_ts'))} ago)"
        else:
            tag = f"\033[32mup\033[0m ({_fmt_age(a.get('alive_ts'))})"
        free = a.get("gpu_free_ids", []) if not stale else []
        total_free += len(free)
        run = a.get("running", [])
        kb = set(a.get("keepbusy_gpus", []))
        free_set = set(free)
        job_by_gpu = {g: r for r in run for g in r.get("gpu_ids", [])}
        gpus = a.get("gpus") or []
        node_mu = sum(g.get("mem_used") or 0 for g in gpus)
        node_mt = sum(g.get("mem_total") or 0 for g in gpus)
        memsum = f"  mem {_fmt_gib(node_mu)}/{_fmt_gib(node_mt)}" if gpus else ""
        print(f"\033[1m{node:<12}\033[0m  {tag}   free {len(free)}/{gtot}{memsum}")
        if gpus:
            for g in gpus:
                i = g.get("index")
                if i in job_by_gpu:
                    r = job_by_gpu[i]
                    lab = f"\033[33mjob {r['id']}\033[0m {r.get('name') or ''}".rstrip()
                elif i in kb:
                    lab = "\033[90mkeepbusy\033[0m"
                elif i in free_set:
                    lab = "\033[32mfree\033[0m"
                else:
                    lab = "\033[31mexternal\033[0m"
                util = g.get("util")
                us = f"{util:3d}%" if isinstance(util, int) else "  ?%"
                mu, mt = g.get("mem_used"), g.get("mem_total")
                ms = (f"{_fmt_gib(mu):>5}/{_fmt_gib(mt)}"
                      if mu is not None and mt else "    ?")
                nb = g.get("procs")
                procs_s = f"  \033[90m{nb}p\033[0m" if nb else ""
                print(f"    gpu{i:<2} {_util_bar(util)} {us}  {ms}  {lab}{procs_s}")
        else:  # heartbeat predates per-GPU stats (agent not yet restarted)
            for r in run:
                print(f"    \033[33mrun\033[0m {r['id']}  gpu "
                      f"{','.join(map(str, r['gpu_ids']))}  {r.get('name') or ''}")
            if kb:
                print(f"    \033[90mkeepbusy on gpu {','.join(map(str, sorted(kb)))}\033[0m")
    print(f"\n\033[1mCLUSTER: {total_free}/{total_gpu} GPUs free\033[0m")


# --------------------------------------------------------------------------- #
# up / down / watch  (bootstrap + supervise agents over ssh, in tmux)
# --------------------------------------------------------------------------- #
TMUX_SESSION = "cowork-agent"
WATCH_SESSION = "cowork-watch"
DEFAULT_WATCH_INTERVAL = 300   # 5 min — safely under the 25-min idle-shutdown


def _target_machines(reg, node):
    if node:
        m = machine_by_name(reg, node)
        if not m:
            sys.exit(f"node {node!r} not in registry")
        return [m]
    return machines(reg)


def _env_prefix(cfg, registry):
    """Env + --registry to pin a tmux-launched process to the same board/registry
    as the caller (tmux's server does not forward arbitrary env vars)."""
    envs = [f"COWORK_STATE={shlex.quote(str(state.state_dir(cfg)))}"]
    if os.environ.get("COWORK_KEEPBUSY"):
        envs.append(f"COWORK_KEEPBUSY={shlex.quote(os.environ['COWORK_KEEPBUSY'])}")
    reg_arg = f" --registry {shlex.quote(registry)}" if registry else ""
    return " ".join(envs), reg_arg


def _tmux_start(machine, session, inner, restart, timeout):
    """Idempotently start `inner` in a tmux session on `machine`.
    Returns one of: 'started' | 'exists' | ('failed', detail)."""
    if restart:
        guard = f"tmux kill-session -t {session} 2>/dev/null; "
    else:
        guard = (f"tmux has-session -t {session} 2>/dev/null && "
                 f"{{ echo EXISTS; exit 0; }}; ")
    remote = guard + f"tmux new-session -d -s {session} {_q(inner)} && echo STARTED"
    try:
        p = probe.run_cmd(machine, remote, timeout=timeout, reuse=True)
    except Exception as e:  # noqa: BLE001
        return ("failed", str(e))
    out = (p.stdout or "").strip()
    if p.returncode != 0:
        err = (p.stderr or p.stdout or "").strip().splitlines()
        return ("failed", err[-1] if err else "?")
    return "exists" if "EXISTS" in out else "started"


def _ensure_up(reg, cfg, targets, restart, timeout):
    """Make sure an agent is running on each target machine. Returns a list of
    (node, action, detail) where action is already|started|exists|failed."""
    state.ensure_dirs(cfg)
    allm = machines(reg)
    if probe.has_remotes(allm):
        probe.ensure_cm_dir()
    agents = {a["node"]: a for a in state.read_agents(cfg)}
    log_dir = state.logs_dir(cfg)
    env_str, reg_arg = _env_prefix(cfg, reg.get("_registry_path"))
    results = []
    for m in targets:
        node = m["name"]
        a = agents.get(node)
        has_hb = a is not None
        fresh = has_hb and not state.agent_is_stale(cfg, a) and not a.get("stopped")
        if fresh and not restart:
            results.append((node, "already", _fmt_age(a.get("alive_ts"))))
            continue
        # Force a kill+restart when a heartbeat exists but is stale/stopped (the
        # agent is dead or hung — clear any lingering tmux session). With no
        # heartbeat at all (never started, or just started and not yet beating)
        # use the gentle has-session guard so we never bounce a starting agent.
        force = restart or (has_hb and not fresh)
        agent_log = log_dir / f"agent-{node}.log"
        inner = (f"exec env {env_str} {ENTRY}{reg_arg} agent --node {node} "
                 f">> {shlex.quote(str(agent_log))} 2>&1")
        r = _tmux_start(m, TMUX_SESSION, inner, force, timeout)
        if isinstance(r, tuple):
            results.append((node, "failed", r[1]))
        else:
            results.append((node, r, None))
    if probe.has_remotes(allm):
        probe.close_masters(allm)
    return results


def _local_machine(reg):
    return next((m for m in machines(reg) if probe.is_local(m)), None)


def _ensure_watch(reg, cfg, interval, timeout, restart=False):
    """Start the supervisor loop in a tmux session on the local (head) node."""
    head = _local_machine(reg)
    if head is None:
        return ("failed", "this host is not in the registry")
    env_str, reg_arg = _env_prefix(cfg, reg.get("_registry_path"))
    watch_log = state.logs_dir(cfg) / "watch.log"
    inner = (f"exec env {env_str} {ENTRY}{reg_arg} watch --interval {interval} "
             f">> {shlex.quote(str(watch_log))} 2>&1")
    return _tmux_start(head, WATCH_SESSION, inner, restart, timeout)


def _print_up_result(node, action, detail):
    if action == "already":
        print(f"{node:<12} already up ({detail})")
    elif action == "exists":
        print(f"{node:<12} already up (tmux session running)")
    elif action == "started":
        print(f"{node:<12} \033[32mstarted\033[0m agent in tmux:{TMUX_SESSION}")
    else:
        print(f"{node:<12} \033[31mFAILED\033[0m: {detail}")


def cmd_up(args, reg, cfg):
    for node, action, detail in _ensure_up(
            reg, cfg, _target_machines(reg, args.node), args.restart, args.timeout):
        _print_up_result(node, action, detail)
    # Auto-start the watchdog on a full `up` (not a single-node op) so agents are
    # kept alive well within the cluster's idle-shutdown window. Opt out: --no-watch.
    if not args.node and not args.no_watch:
        r = _ensure_watch(reg, cfg, args.watch_interval, args.timeout)
        if r == "started":
            print(f"{'watchdog':<12} \033[32mstarted\033[0m (every {args.watch_interval}s) "
                  f"in tmux:{WATCH_SESSION}")
        elif r == "exists":
            print(f"{'watchdog':<12} already running (tmux:{WATCH_SESSION})")
        else:
            print(f"{'watchdog':<12} \033[31mFAILED\033[0m: {r[1] if isinstance(r, tuple) else r}")
    print("\nagents publish heartbeats to state/agents/ — check `cowork status`")


def cmd_watch(args, reg, cfg):
    import signal
    stop = {"v": False}

    def _onsig(_s, _f):
        stop["v"] = True
    for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, _onsig)
    interval = args.interval
    print(f"[watch] supervising agents every {interval:g}s — re-ups any that go "
          f"down (idle-shutdown safety net). Ctrl-C to stop.", flush=True)
    while not stop["v"]:
        try:
            results = _ensure_up(reg, cfg, machines(reg), restart=False,
                                 timeout=args.timeout)
            changed = [(n, a, d) for n, a, d in results if a in ("started", "failed")]
            if changed:
                for n, a, d in changed:
                    print(f"[watch {state.now_iso()}] {n}: {a}"
                          f"{(' ' + str(d)) if d else ''}", flush=True)
        except Exception as e:  # noqa: BLE001 - never let one tick kill the watcher
            print(f"[watch {state.now_iso()}] error: {e!r}", flush=True)
        if args.once:
            break
        slept = 0.0
        while slept < interval and not stop["v"]:
            time.sleep(min(1.0, interval - slept))
            slept += 1.0
    print("[watch] stopped", flush=True)


def cmd_down(args, reg, cfg):
    if probe.has_remotes(machines(reg)):
        probe.ensure_cm_dir()
    # A full down must stop the watchdog FIRST, or it would re-up the agents we
    # are about to stop. Single-node down leaves the watchdog alone (and will be
    # revived by it within the interval — use a full `down` to stop everything).
    if not args.node:
        head = _local_machine(reg)
        if head is not None:
            probe.run_cmd(head, f"tmux kill-session -t {WATCH_SESSION} 2>/dev/null; true",
                          timeout=args.timeout, reuse=True)
            print(f"{'watchdog':<12} stopped")
    for m in _target_machines(reg, args.node):
        node = m["name"]
        remote = f"tmux kill-session -t {TMUX_SESSION} 2>/dev/null && echo killed || echo none"
        try:
            p = probe.run_cmd(m, remote, timeout=args.timeout, reuse=True)
            out = (p.stdout or "").strip()
            print(f"{node:<12} {'stopped' if out == 'killed' else 'no agent running'}")
        except Exception as e:  # noqa: BLE001
            print(f"{node:<12} \033[31mFAILED\033[0m: {e}")
    if probe.has_remotes(machines(reg)):
        probe.close_masters(machines(reg))
    if args.node:
        print("note: the watchdog (if running) will re-up this node within its "
              "interval; use a full `cowork down` to stop everything.")
    print("(running jobs are detached and keep running)")


def _q(s):
    return "'" + s.replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------- #
# agent (the daemon itself)
# --------------------------------------------------------------------------- #
def cmd_agent(args, reg, cfg):
    agent_mod.Agent(reg, cfg, agent_mod.resolve_node(reg, args.node)).run()


# --------------------------------------------------------------------------- #
# arg parsing
# --------------------------------------------------------------------------- #
def build_parser():
    ap = argparse.ArgumentParser(prog="cowork",
                                 description="file-queue job scheduling across the dev cluster")
    ap.add_argument("--registry", default=None, help="path to machines.yaml")
    sub = ap.add_subparsers(dest="command", required=True)

    s = sub.add_parser("submit", help="queue a job and return immediately")
    s.add_argument("--env", default="base", help="conda env to activate (default: base)")
    s.add_argument("--gpus", type=int, default=1, help="number of GPUs to request (default: 1; 0 = CPU)")
    s.add_argument("--gpu-ids", default=None, help="specific physical GPU ids, e.g. 0,1 (overrides --gpus)")
    s.add_argument("--node", default=None, help="pin to a specific node (default: any that fits)")
    s.add_argument("--name", default=None, help="human label for the job")
    s.add_argument("--cwd", default=None, help="working directory (default: current dir)")
    s.add_argument("--conda-root", default=None, help="conda root (default: registry's first)")
    s.add_argument("--env-var", action="append", metavar="K=V", help="extra env var (repeatable)")
    s.add_argument("--priority", type=int, default=0, help="reserved for future ordering")
    s.add_argument("-q", "--quiet", action="store_true", help="print only the job id")
    s.add_argument("cmd", nargs=argparse.REMAINDER, help="-- COMMAND ARGS...")

    s = sub.add_parser("ps", help="list jobs")
    s.add_argument("--all", "-a", action="store_true", help="include done/failed")

    s = sub.add_parser("logs", help="show a job's output")
    s.add_argument("id")
    s.add_argument("-f", "--follow", action="store_true", help="stream new output")
    s.add_argument("--err", action="store_true", help="show stderr instead of stdout")

    s = sub.add_parser("cancel", help="cancel a queued or running job")
    s.add_argument("id")

    sub.add_parser("status", help="cluster view from agent heartbeats (no SSH)")

    s = sub.add_parser("up", help="start agents on the nodes + the watchdog (idempotent)")
    s.add_argument("--node", default=None, help="only this node (skips the watchdog)")
    s.add_argument("--restart", action="store_true", help="restart even if already up")
    s.add_argument("--no-watch", action="store_true", help="do not start the watchdog")
    s.add_argument("--watch-interval", type=int, default=DEFAULT_WATCH_INTERVAL,
                   metavar="SEC", help=f"watchdog re-up interval (default: {DEFAULT_WATCH_INTERVAL}s)")
    s.add_argument("--timeout", type=float, default=10)

    s = sub.add_parser("down", help="stop agents + watchdog (running jobs keep running)")
    s.add_argument("--node", default=None, help="only this node (leaves watchdog running)")
    s.add_argument("--timeout", type=float, default=10)

    s = sub.add_parser("watch", help="supervisor loop: re-up any agent that goes down")
    s.add_argument("--interval", type=int, default=DEFAULT_WATCH_INTERVAL,
                   metavar="SEC", help=f"seconds between checks (default: {DEFAULT_WATCH_INTERVAL})")
    s.add_argument("--once", action="store_true", help="run one check and exit")
    s.add_argument("--timeout", type=float, default=10)

    s = sub.add_parser("agent", help="run the per-node daemon (used by `up`)")
    s.add_argument("--node", default=None, help="node name (default: auto-detect by hostname)")

    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    # `submit -- a b c` leaves a leading "--" in REMAINDER; drop it.
    if getattr(args, "cmd", None) and args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    reg = load_registry(args.registry) if args.registry else load_registry()
    # remember the registry path so `up`/`watch` can pin tmux-launched children
    # to the same registry (None => the default machines.yaml next to the package)
    reg["_registry_path"] = args.registry
    cfg = state.load_config(reg)
    handlers = {
        "submit": cmd_submit, "ps": cmd_ps, "logs": cmd_logs, "cancel": cmd_cancel,
        "status": cmd_status, "up": cmd_up, "down": cmd_down, "watch": cmd_watch,
        "agent": cmd_agent,
    }
    handlers[args.command](args, reg, cfg)
