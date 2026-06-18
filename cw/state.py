"""The /share coordination board: paths, atomic file ops, config, job + agent I/O.

Everything lives under <repo>/state on the shared NFS so any node sees it without
SSH. Job specs move between state dirs by atomic os.rename — on NFSv4 a rename is
atomic on the server, so a decentralized claim ("rename queued/<id> -> running/")
has exactly one winner, no locks. Atomic *writes* go through a temp file in the
same directory followed by os.replace, so a reader never sees a half-written file.
"""
import json
import os
import secrets
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Config (optional `cowork:` block in machines.yaml; everything has a default)
# --------------------------------------------------------------------------- #
DEFAULTS = {
    "state_dir": "state",
    "heartbeat_interval": 5,      # seconds between agent heartbeats
    "poll_interval": 3,           # seconds between agent claim/reap cycles
    "mem_thresh": 2048,           # MiB; a GPU at/under this with no procs is free
    "heartbeat_stale_factor": 4,  # heartbeat older than factor*interval => DOWN
    "keepbusy": {
        "enabled": True,
        "env": "base",
        "run_py": "/share/ziyangrao/keepbusy/run.py",
        "idle_minutes": 0.5,
    },
}


def _merge(base, over):
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def load_config(reg):
    cfg = _merge(DEFAULTS, reg.get("cowork") or {})
    # operational override: COWORK_KEEPBUSY=0/off/false disables keepbusy supervision
    if os.environ.get("COWORK_KEEPBUSY", "").lower() in ("0", "off", "false", "no"):
        cfg["keepbusy"]["enabled"] = False
    return cfg


def state_dir(cfg):
    sd = os.environ.get("COWORK_STATE") or cfg.get("state_dir", "state")
    p = Path(sd)
    return p if p.is_absolute() else (REPO / p)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
# Job states are directories; a job is a single <id>.json that moves between them.
JOB_STATES = ("queued", "running", "done", "failed")


def _d(cfg, *parts):
    return state_dir(cfg).joinpath(*parts)


def jobs_dir(cfg, state):
    return _d(cfg, "jobs", state)


def agents_dir(cfg):
    return _d(cfg, "agents")


def logs_dir(cfg):
    return _d(cfg, "logs")


def ensure_dirs(cfg):
    for s in JOB_STATES:
        jobs_dir(cfg, s).mkdir(parents=True, exist_ok=True)
    agents_dir(cfg).mkdir(parents=True, exist_ok=True)
    logs_dir(cfg).mkdir(parents=True, exist_ok=True)


def job_path(cfg, state, job_id):
    return jobs_dir(cfg, state) / f"{job_id}.json"


def cancel_marker(cfg, job_id):
    # A touch-file any node can drop to ask the owning agent to stop a running job.
    return jobs_dir(cfg, "running") / f"{job_id}.cancel"


def agent_path(cfg, node):
    return agents_dir(cfg) / f"{node}.json"


def keepbusy_pid_file(cfg, node):
    # Tracks the agent's keepbusy pid so a restarted agent can reap an orphan
    # (keepbusy disguises its process name, so it can't be found by name).
    return agents_dir(cfg) / f"{node}.keepbusy.pid"


def log_paths(cfg, job_id):
    d = logs_dir(cfg)
    return {
        "out": d / f"{job_id}.out",
        "err": d / f"{job_id}.err",
        "rc": d / f"{job_id}.rc",
    }


# --------------------------------------------------------------------------- #
# Atomic primitives
# --------------------------------------------------------------------------- #
def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)          # atomic within the same directory


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, ValueError):
        return None


def claim(src, dst):
    """Atomically move src -> dst. Returns True if we won the rename.

    A FileNotFoundError means another agent already claimed it (the source
    vanished) — we lost the race, which is a normal outcome, not an error.
    """
    try:
        os.rename(src, dst)
        return True
    except (FileNotFoundError, NotADirectoryError):
        return False


# --------------------------------------------------------------------------- #
# Ids, time, job enumeration
# --------------------------------------------------------------------------- #
def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def new_job_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def iter_jobs(cfg, states=JOB_STATES):
    """Yield (state, job_dict) for every job spec in the given states."""
    for s in states:
        d = jobs_dir(cfg, s)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            j = read_json(p)
            if j is not None:
                j["_state"] = s
                j["_path"] = str(p)
                yield s, j


def find_job(cfg, ref, states=JOB_STATES):
    """Resolve a full id or unique prefix to (state, job_dict). None if absent,
    "ambiguous" if the prefix matches more than one job."""
    hits = [(s, j) for s, j in iter_jobs(cfg, states)
            if j.get("id") == ref or j.get("id", "").startswith(ref)]
    exact = [(s, j) for s, j in hits if j.get("id") == ref]
    if exact:
        return exact[0]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        return "ambiguous"
    return None


def read_agents(cfg):
    out = []
    d = agents_dir(cfg)
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            a = read_json(p)
            if a is not None:
                out.append(a)
    return out


def agent_is_stale(cfg, agent):
    interval = cfg.get("heartbeat_interval", 5)
    factor = cfg.get("heartbeat_stale_factor", 4)
    age = time.time() - agent.get("alive_ts", 0)
    return age > interval * factor
