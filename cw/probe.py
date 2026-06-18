"""nvidia-smi probing + ssh helpers, shared by the GPU monitor, the agent
(local probe to spot *external* GPU jobs) and `cowork up/down` (remote exec).

Remote probes reuse one persistent SSH connection per host (OpenSSH
ControlMaster/ControlPersist, sockets under /tmp/cowork-ssh-<uid>/), so repeated
probes don't pay a fresh handshake. Standard library only.
"""
import os
import socket
import subprocess
import tempfile
from pathlib import Path

# Probe command: GPU table, compute-apps, then ps to recover full command lines
# (nvidia-smi only reports a truncated process_name like "python").
REMOTE_CMD = (
    "echo @@GPU@@; "
    "nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu "
    "--format=csv,noheader,nounits; "
    "echo @@APP@@; "
    "nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory,process_name "
    "--format=csv,noheader,nounits; "
    "echo @@PS@@; "
    "__p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits "
    "| sort -un | paste -sd, -); "
    "if [ -n \"$__p\" ]; then ps -o pid=,user=,etime=,args= -p \"$__p\" || true; fi"
)

CM_DIR = Path(tempfile.gettempdir()) / f"cowork-ssh-{os.getuid()}"
_CM_PERSIST = 30  # seconds the master stays alive after the last connection


def is_local(machine):
    return machine.get("hostname") == socket.gethostname()


def has_remotes(machines):
    return any(not is_local(m) for m in machines)


def ensure_cm_dir():
    os.makedirs(CM_DIR, exist_ok=True)
    os.chmod(CM_DIR, 0o700)


def _cm_opts():
    return [
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={CM_DIR}/%C",
        "-o", f"ControlPersist={_CM_PERSIST}",
    ]


def _ssh_target(machine):
    ssh = machine.get("ssh", {}) or {}
    return ssh.get("host"), str(ssh.get("port", 22)), ssh.get("user", "root")


def ssh_argv(machine, timeout, reuse=True):
    """The ssh command prefix (no remote command appended) for a machine."""
    host, port, user = _ssh_target(machine)
    argv = [
        "ssh", "-p", port,
        "-o", "BatchMode=yes",
        "-o", f"ConnectTimeout={int(timeout)}",
        "-o", "StrictHostKeyChecking=accept-new",
    ]
    if reuse:
        argv += _cm_opts()
    argv += [f"{user}@{host}"]
    return argv


def run_cmd(machine, command, timeout=10, reuse=True):
    """Run a shell command on a machine — locally if it's our node, else via ssh.

    Returns the CompletedProcess (caller inspects returncode/stdout/stderr).
    """
    if is_local(machine):
        argv = ["bash", "-lc", command]
    else:
        argv = ssh_argv(machine, timeout, reuse) + [command]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout + 5)


def _run(machine, timeout, reuse=True):
    p = run_cmd(machine, REMOTE_CMD, timeout, reuse)
    if p.returncode != 0:
        raise RuntimeError((p.stderr or p.stdout or "ssh failed").strip().splitlines()[-1])
    return p.stdout


def close_masters(machines):
    """Best-effort teardown of any ControlMaster sockets we opened."""
    for m in machines:
        if is_local(m):
            continue
        host, port, user = _ssh_target(m)
        if not host:
            continue
        argv = ["ssh", "-p", port, "-o", f"ControlPath={CM_DIR}/%C",
                "-O", "exit", f"{user}@{host}"]
        try:
            subprocess.run(argv, capture_output=True, text=True, timeout=5)
        except Exception:  # noqa: BLE001 - cleanup is best-effort
            pass


def _parse_smi(stdout):
    marks = {"@@GPU@@": "gpu", "@@APP@@": "app", "@@PS@@": "ps"}
    sections = {"gpu": [], "app": [], "ps": []}
    section = None
    for line in stdout.splitlines():
        s = line.strip()
        if s in marks:
            section = marks[s]; continue
        if not s or section is None:
            continue
        sections[section].append(line.rstrip())

    ps_by_pid = {}
    for line in sections["ps"]:
        parts = line.split(None, 3)
        if len(parts) < 3:
            continue
        pid, user, etime = parts[0], parts[1], parts[2]
        cmd = parts[3] if len(parts) > 3 else ""
        if "-" in etime and etime.split("-", 1)[0].isdigit() and int(etime.split("-", 1)[0]) > 9999:
            etime = "?"
        try:
            ps_by_pid[int(pid)] = {"user": user, "runtime": etime, "cmd": cmd}
        except ValueError:
            continue

    procs_by_uuid = {}
    for line in sections["app"]:
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 3 or not parts[0]:
            continue
        uuid = parts[0]
        try:
            pid, gpu_mem = int(parts[1]), int(parts[2])
        except ValueError:
            continue
        name = ",".join(parts[3:]).strip() if len(parts) > 3 else ""
        if name in ("[Not Found]", "[Insufficient Permissions]"):
            name = ""
        info = ps_by_pid.get(pid)
        procs_by_uuid.setdefault(uuid, []).append({
            "pid": pid,
            "gpu_mem": gpu_mem,
            "name": name,
            "user": info["user"] if info else "?",
            "runtime": info["runtime"] if info else "?",
            "cmd": (info["cmd"] if info and info["cmd"] else name),
            "resolved": bool(info),
        })

    gpus = []
    for line in sections["gpu"]:
        idx, uuid, mem_used, mem_total, util = [c.strip() for c in line.split(",")]
        plist = sorted(procs_by_uuid.get(uuid, []), key=lambda p: -p["gpu_mem"])
        gpus.append({
            "index": int(idx),
            "uuid": uuid,
            "mem_used": int(mem_used),
            "mem_total": int(mem_total),
            "util": int(util),
            "procs": len(plist),
            "proc_list": plist,
        })
    return gpus


def probe(machine, timeout, mem_thresh, reuse=True):
    name = machine.get("name", machine.get("hostname", "?"))
    res = {"name": name, "hostname": machine.get("hostname"), "ok": False,
           "error": None, "gpus": []}
    try:
        gpus = _parse_smi(_run(machine, timeout, reuse))
        for g in gpus:
            g["free"] = g["procs"] == 0 and g["mem_used"] <= mem_thresh
        res["gpus"] = gpus
        res["ok"] = True
    except Exception as e:  # noqa: BLE001 - surface any failure per node
        res["error"] = str(e)
    return res
