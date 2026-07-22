"""The per-node cowork agent.

One of these runs on every node (started once via `cowork up`, lives in a tmux
session). It is the single authority for its node's GPUs:

  * watches state/jobs/queued/ and claims any job it can fit, via an atomic
    rename (queued/<id>.json -> running/<id>.json) — exactly one agent wins;
  * pins CUDA_VISIBLE_DEVICES, activates the job's conda env, launches it
    DETACHED (setsid) so the job survives an agent restart/redeploy, and streams
    stdout/stderr to state/logs/;
  * runs keepbusy on whatever GPUs are currently free and SIGTERMs it the instant
    a real job needs those GPUs — folding keepbusy into cowork instead of letting
    it guess from nvidia-smi;
  * publishes a heartbeat (state/agents/<node>.json) so `cowork status` shows the
    cluster without any SSH.

A job's bash wrapper writes its exit code to state/logs/<id>.rc, so completion and
exit status are recoverable even across an agent restart (the re-adopted job is no
longer our child, so we can't waitpid it).
"""
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from . import probe, state
from .registry import load_registry, machine_by_name, machines


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True            # exists but owned by someone else => alive
    return True


class Agent:
    def __init__(self, reg, cfg, node):
        self.reg = reg
        self.cfg = cfg
        self.node = node
        self.machine = machine_by_name(reg, node)
        if self.machine is None:
            raise SystemExit(f"node {node!r} not found in registry")
        self.hostname = self.machine.get("hostname")
        gpus = self.machine.get("gpus") or {}
        self.gpu_ids = list(range(int(gpus.get("count", 0) or 0)))
        self.conda_root_default = (reg.get("conda_roots") or
                                   ["/share/ziyangrao/miniconda3"])[0]
        self.mem_thresh = cfg.get("mem_thresh", 2048)
        self.poll = cfg.get("poll_interval", 3)
        self.kb = cfg.get("keepbusy", {}) or {}
        # owned[id] = {spec, popen|None, pid, pgid, gpu_ids}
        self.owned = {}
        self.keepbusy_proc = None
        self.keepbusy_gpus = []
        self._gpu_stats = {}        # idx -> last probe row (util/mem/procs), for heartbeat
        self._stop = False

    # ------------------------------------------------------------------ free
    def _compute_free(self):
        """GPU ids available to claim: total minus cowork-placed jobs minus any
        GPU an EXTERNAL (non-cowork, non-our-keepbusy) process is using.

        keepbusy GPUs stay free-to-claim (keepbusy is preemptible). The hard part
        is telling our OWN keepbusy apart from a real external job sharing one of
        its GPUs. We can't do it by pid inside a PID-isolated pod: nvidia-smi
        reports every GPU process under its HOST pid, but keepbusy runs in our pod
        and we only know its POD pid — they never match, and `ps` in the pod can't
        resolve the host pid either (so the proc comes back `resolved=False`). A
        naive "any foreign pid => external" test therefore mistakes keepbusy for an
        external job every time it loads a GPU, drops that GPU, and makes
        _set_keepbusy() thrash keepbusy in a restart loop — it never sustains load,
        the GPUs sit at 0% util, and the node gets reclaimed by the cluster's
        idle-shutdown.

        We exploit keepbusy's own shape: it is a single DataParallel process, so on
        a GPU it owns it contributes AT MOST ONE compute-app entry (and zero while
        it is merely idle-monitoring, before it has built a CUDA context). So on a
        keepbusy GPU we exclude it (treat as externally-busy) when ANY of:
          (a) a *resolvable* foreign proc is present — a genuine external job on a
              bare host like `head`, where pids resolve;
          (b) there are >= 2 compute procs — keepbusy alone is <= 1, so a second
              proc is someone else. This is the SAME threshold keepbusy itself
              yields on (it returns to monitoring when it sees >= 2 compute pids),
              so the two stay consistent and keepbusy is never restart-thrashed by
              its own single proc; and it closes the blind spot where an external
              job's unresolvable sibling-container proc used to be silently ignored;
          (c) there is real memory in use with ZERO compute procs — keepbusy holding
              a GPU always shows its own proc, so mem-without-a-proc is leftover /
              untracked external usage, not us.
        Residual (accepted, not fixable inside the pod): if keepbusy is still
        idle-monitoring (0 procs) when an external job arrives and pins util so
        keepbusy never builds a context, that job shows as exactly ONE unresolvable
        proc — indistinguishable from keepbusy actively loading alone (also one
        unresolvable proc). We can't disambiguate a host pid we can't see, so we err
        toward NOT thrashing keepbusy; keepbusy yields to that job on its own, and
        the window is narrow (external must land inside keepbusy's ~30s warm-up).

        On a GPU we are NOT keeping busy, any compute proc — resolvable or not —
        still owns the GPU, so we exclude it as before."""
        cowork_used = {g for j in self.owned.values() for g in j["gpu_ids"]}
        kb = set(self.keepbusy_gpus)
        kb_pids = {self.keepbusy_proc.pid} if self.keepbusy_proc is not None else set()
        res = probe.probe(self.machine, timeout=5, mem_thresh=self.mem_thresh,
                          reuse=False)
        gstate = {g["index"]: g for g in res["gpus"]} if res["ok"] else {}
        self._gpu_stats = gstate    # publish util/mem in the next heartbeat
        free = []
        for idx in self.gpu_ids:
            if idx in cowork_used:
                continue
            g = gstate.get(idx)
            if g is not None:
                foreign = [p for p in g["proc_list"] if p["pid"] not in kb_pids]
                if idx in kb:
                    # GPU running our keepbusy: exclude on a resolvable foreign
                    # proc, on >= 2 procs (keepbusy alone is <= 1), or on real
                    # memory with no proc at all (keepbusy always shows its proc
                    # when it holds a GPU) — see docstring.
                    if (any(p["resolved"] for p in foreign)
                            or len(g["proc_list"]) >= 2
                            or (not g["proc_list"]
                                and g["mem_used"] > self.mem_thresh)):
                        continue
                else:
                    # any compute proc that isn't our keepbusy => external job
                    if foreign:
                        continue
                    # leftover/untracked memory on a GPU we are NOT keeping busy
                    if g["mem_used"] > self.mem_thresh:
                        continue
            free.append(idx)
        return free

    def _fit(self, spec, free):
        if spec.get("node") and spec["node"] != self.node:
            return None
        want_ids = spec.get("gpu_ids")
        if want_ids:
            return list(want_ids) if all(i in free for i in want_ids) else None
        n = int(spec.get("gpus", 1) or 0)
        if n == 0:
            return []
        return free[:n] if len(free) >= n else None

    # ---------------------------------------------------------------- launch
    def _conda_root(self, spec):
        return spec.get("conda_root") or self.conda_root_default

    def _launch(self, spec, gpu_ids):
        jid = spec["id"]
        lp = state.log_paths(self.cfg, jid)
        for p in lp.values():
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        conda_root = self._conda_root(spec)
        env_name = spec.get("env") or "base"
        cwd = spec.get("cwd") or str(state.REPO)
        inner = shlex.join(spec["cmd"])
        script = (
            "set +e\n"
            f"source {shlex.quote(conda_root)}/etc/profile.d/conda.sh\n"
            f"conda activate {shlex.quote(env_name)}\n"
            f"cd {shlex.quote(cwd)}\n"
            f"( {inner} )\n"
            "rc=$?\n"
            f"echo $rc > {shlex.quote(str(lp['rc']))}\n"
            "exit $rc\n"
        )
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["COWORK_JOB_ID"] = jid
        for k, v in (spec.get("env_vars") or {}).items():
            env[str(k)] = str(v)
        try:
            out = open(lp["out"], "ab")
            err = open(lp["err"], "ab")
            p = subprocess.Popen(
                ["bash", "-lc", script],
                cwd=cwd, env=env, stdout=out, stderr=err,
                stdin=subprocess.DEVNULL, start_new_session=True,
            )
        except Exception as e:  # noqa: BLE001
            log(f"launch FAILED job {jid}: {e}")
            self._finalize_spec(spec, gpu_ids, "failed", None,
                                reason=f"launch error: {e}")
            return False
        self.owned[jid] = {"spec": spec, "popen": p, "pid": p.pid,
                           "pgid": p.pid, "gpu_ids": list(gpu_ids)}
        runtime = dict(spec)
        runtime.update({
            "assigned_node": self.node, "assigned_gpu_ids": list(gpu_ids),
            "pid": p.pid, "pgid": p.pid, "started_at": state.now_iso(),
            "started_ts": time.time(),
            "log_out": str(lp["out"]), "log_err": str(lp["err"]),
            "rc_file": str(lp["rc"]),
        })
        state.atomic_write_json(state.job_path(self.cfg, "running", jid), runtime)
        log(f"launched job {jid} on gpu(s) {gpu_ids} env={env_name} pid={p.pid}: {inner}")
        return True

    # ------------------------------------------------------------------ reap
    def _job_exit_code(self, j):
        lp = state.log_paths(self.cfg, j["spec"]["id"])
        if lp["rc"].exists():
            try:
                return int(lp["rc"].read_text().strip())
            except (ValueError, OSError):
                return None
        popen = j.get("popen")
        if popen is not None and popen.poll() is not None:
            return popen.returncode
        return None

    def _is_finished(self, j):
        lp = state.log_paths(self.cfg, j["spec"]["id"])
        if lp["rc"].exists():
            return True
        popen = j.get("popen")
        if popen is not None:
            return popen.poll() is not None
        return not _pid_alive(j.get("pid"))

    def _reap(self):
        for jid in list(self.owned):
            j = self.owned[jid]
            if not self._is_finished(j):
                continue
            code = self._job_exit_code(j)
            st = "done" if code == 0 else "failed"
            reason = None if code == 0 else (
                f"exit {code}" if code is not None else "process exited (code unknown)")
            self._finalize(jid, st, code, reason)

    def _finalize(self, jid, st, code, reason):
        j = self.owned.pop(jid, None)
        spec = j["spec"] if j else None
        gpu_ids = j["gpu_ids"] if j else []
        self._finalize_spec(spec, gpu_ids, st, code, reason, jid=jid)
        log(f"job {jid} -> {st} (code={code}{', ' + reason if reason else ''})")

    def _finalize_spec(self, spec, gpu_ids, st, code, reason, jid=None):
        jid = jid or (spec or {}).get("id")
        run_p = state.job_path(self.cfg, "running", jid)
        rec = state.read_json(run_p) or dict(spec or {})
        rec.update({
            "assigned_node": self.node, "assigned_gpu_ids": list(gpu_ids),
            "finished_at": state.now_iso(), "finished_ts": time.time(),
            "exit_code": code, "reason": reason,
        })
        state.atomic_write_json(state.job_path(self.cfg, st, jid), rec)
        for p in (run_p, state.cancel_marker(self.cfg, jid)):
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    # ---------------------------------------------------------------- cancel
    def _handle_cancels(self):
        for jid in list(self.owned):
            marker = state.cancel_marker(self.cfg, jid)
            if not marker.exists():
                continue
            j = self.owned[jid]
            log(f"cancel requested for {jid}; terminating pgid {j['pgid']}")
            self._kill_group(j["pgid"])
            self._finalize(jid, "failed", None, "cancelled")

    def _kill_group(self, pgid):
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pgid, sig)
            except ProcessLookupError:
                return
            for _ in range(20):
                try:
                    os.killpg(pgid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.1)

    # -------------------------------------------------------------- keepbusy
    def _kb_python(self):
        env_name = self.kb.get("env", "base")
        root = self.conda_root_default
        if env_name == "base":
            return f"{root}/bin/python"
        return f"{root}/envs/{env_name}/bin/python"

    def _stop_keepbusy(self):
        p = self.keepbusy_proc
        self.keepbusy_proc = None
        self.keepbusy_gpus = []
        try:
            state.keepbusy_pid_file(self.cfg, self.node).unlink()
        except FileNotFoundError:
            pass
        if p is None or p.poll() is not None:
            return
        try:
            p.terminate()
            for _ in range(80):
                if p.poll() is not None:
                    return
                time.sleep(0.1)
            p.kill()
        except Exception:  # noqa: BLE001
            pass

    def _reap_orphan_keepbusy(self):
        """Kill a keepbusy left over from a previous agent that crashed (it
        disguises its name, so we find it only via the recorded pid)."""
        pf = state.keepbusy_pid_file(self.cfg, self.node)
        rec = state.read_json(pf)
        if not rec:
            return
        pid = rec.get("pid")
        if pid and _pid_alive(pid):
            log(f"reaping orphaned keepbusy pid={pid} from a previous agent")
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(80):
                    if not _pid_alive(pid):
                        break
                    time.sleep(0.1)
                if _pid_alive(pid):
                    os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        try:
            pf.unlink()
        except FileNotFoundError:
            pass

    def _set_keepbusy(self, free):
        desired = sorted(free)
        if (desired == sorted(self.keepbusy_gpus)
                and self.keepbusy_proc is not None
                and self.keepbusy_proc.poll() is None):
            return
        self._stop_keepbusy()
        if not self.kb.get("enabled", True) or not desired:
            return
        run_py = self.kb.get("run_py", "/share/ziyangrao/keepbusy/run.py")
        if not Path(run_py).exists():
            return
        cmd = [
            self._kb_python(), run_py,
            "--gpus", ",".join(str(g) for g in desired),
            "--idle-minutes", str(self.kb.get("idle_minutes", 0.5)),
            "--poll-interval", "1",
        ]
        kb_log = state.logs_dir(self.cfg) / f"keepbusy-{self.node}.log"
        try:
            f = open(kb_log, "ab")
            self.keepbusy_proc = subprocess.Popen(
                cmd, stdout=f, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL)
            self.keepbusy_gpus = desired
            state.atomic_write_json(state.keepbusy_pid_file(self.cfg, self.node),
                                    {"pid": self.keepbusy_proc.pid, "gpus": desired})
            log(f"keepbusy running on gpu(s) {desired}")
        except Exception as e:  # noqa: BLE001
            log(f"keepbusy launch failed: {e}")

    # -------------------------------------------------------------- lifecycle
    def reconcile(self):
        """On startup, recover the running/ dir: re-adopt our still-alive jobs,
        finalize ones that finished while we were down, requeue interrupted
        claims."""
        for j in list(state.jobs_dir(self.cfg, "running").glob("*.json")):
            rec = state.read_json(j)
            if rec is None:
                continue
            jid = rec.get("id") or j.stem
            owner = rec.get("assigned_node")
            pid = rec.get("pid")
            lp = state.log_paths(self.cfg, jid)
            if owner != self.node:
                continue
            if pid is None:
                # claimed but never launched (agent crashed mid-claim) -> retry
                state.atomic_write_json(state.job_path(self.cfg, "queued", jid),
                                        {k: v for k, v in rec.items()
                                         if not k.startswith("assigned")})
                try:
                    j.unlink()
                except FileNotFoundError:
                    pass
                log(f"requeued interrupted claim {jid}")
                continue
            if lp["rc"].exists():
                try:
                    code = int(lp["rc"].read_text().strip())
                except (ValueError, OSError):
                    code = None
                self._finalize_spec(rec, rec.get("assigned_gpu_ids", []),
                                    "done" if code == 0 else "failed", code,
                                    None if code == 0 else f"exit {code}", jid=jid)
                log(f"reaped completed-while-down job {jid} (code={code})")
            elif _pid_alive(pid):
                self.owned[jid] = {"spec": rec, "popen": None, "pid": pid,
                                   "pgid": rec.get("pgid", pid),
                                   "gpu_ids": rec.get("assigned_gpu_ids", [])}
                log(f"re-adopted running job {jid} pid={pid} "
                    f"gpu(s) {rec.get('assigned_gpu_ids')}")
            else:
                self._finalize_spec(rec, rec.get("assigned_gpu_ids", []),
                                    "failed", None,
                                    "process gone (agent was down)", jid=jid)
                log(f"job {jid} died while agent was down -> failed")

    def _heartbeat(self, free, stopped=False):
        rec = {
            "node": self.node, "hostname": self.hostname, "agent_pid": os.getpid(),
            "alive_ts": time.time(), "alive_at": state.now_iso(),
            "gpu_total": len(self.gpu_ids), "gpu_free_ids": sorted(free),
            "keepbusy_gpus": sorted(self.keepbusy_gpus),
            "gpus": [{"index": i, "util": g.get("util"),
                      "mem_used": g.get("mem_used"), "mem_total": g.get("mem_total"),
                      "procs": g.get("procs")}
                     for i, g in sorted(self._gpu_stats.items())],
            "running": [{"id": jid, "name": j["spec"].get("name"),
                         "gpu_ids": j["gpu_ids"]} for jid, j in self.owned.items()],
            "poll_interval": self.poll, "stopped": stopped, "version": 1,
        }
        state.atomic_write_json(state.agent_path(self.cfg, self.node), rec)

    def _claim_cycle(self, free):
        for p in sorted(state.jobs_dir(self.cfg, "queued").glob("*.json")):
            spec = state.read_json(p)
            if spec is None:
                continue
            pick = self._fit(spec, free)
            if pick is None:
                continue
            jid = spec.get("id") or p.stem
            spec["id"] = jid
            if not state.claim(p, state.job_path(self.cfg, "running", jid)):
                continue                      # another agent won the rename
            if set(pick) & set(self.keepbusy_gpus):
                self._stop_keepbusy()         # release GPUs before the real job
            if self._launch(spec, pick):
                free = [g for g in free if g not in pick]
        return free

    def run(self):
        state.ensure_dirs(self.cfg)
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._on_signal)
        log(f"agent up: node={self.node} host={self.hostname} "
            f"gpus={self.gpu_ids} state={state.state_dir(self.cfg)}")
        self._reap_orphan_keepbusy()
        self.reconcile()
        while not self._stop:
            try:
                self._reap()
                self._handle_cancels()
                free = self._compute_free()
                free = self._claim_cycle(free)
                self._reap()
                free = self._compute_free()
                self._set_keepbusy(free)
                self._heartbeat(free)
            except Exception as e:  # noqa: BLE001 - never let one cycle kill the agent
                log(f"cycle error: {e!r}")
            for _ in range(int(self.poll * 10)):
                if self._stop:
                    break
                time.sleep(0.1)
        self._shutdown()

    def _on_signal(self, signum, _frame):
        log(f"received {signal.Signals(signum).name}; shutting down "
            f"(running jobs are detached and keep running)")
        self._stop = True

    def _shutdown(self):
        self._stop_keepbusy()
        try:
            self._heartbeat(self._compute_free(), stopped=True)
        except Exception:  # noqa: BLE001
            pass
        log("agent stopped")


def resolve_node(reg, node_arg):
    if node_arg:
        return node_arg
    host = socket.gethostname()
    for m in machines(reg):
        if m.get("hostname") == host:
            return m.get("name")
    raise SystemExit(
        f"could not auto-detect node: hostname {host!r} not in registry; "
        f"pass --node NAME")


def main(node_arg=None, registry=None):
    reg = load_registry(registry) if registry else load_registry()
    cfg = state.load_config(reg)
    node = resolve_node(reg, node_arg)
    Agent(reg, cfg, node).run()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
