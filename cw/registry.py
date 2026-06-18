"""Minimal YAML-subset parser for machines.yaml — standard library only.

Handles the structure used in machines.yaml: nested maps, lists of maps, scalar
lists, top-level &anchors and *aliases (aliases are left unresolved — callers
that need conda_* fields read them off the anchor entries directly). This is the
same parser that backed gpus.py, factored out so the scheduler/agent/monitor all
share one registry loader.
"""
from pathlib import Path

REG = Path(__file__).resolve().parent.parent / "machines.yaml"
# Private per-node connection details (hostname / ssh host+port+user / internal_ip)
# live here, NOT in the tracked machines.yaml, and this file is gitignored. See
# machines.local.example.yaml for the schema. Override the path with COWORK_LOCAL.
LOCAL = Path(__file__).resolve().parent.parent / "machines.local.yaml"


def _scalar(v):
    v = v.strip()
    if (v[:1], v[-1:]) in (('"', '"'), ("'", "'")):
        return v[1:-1]           # quoted -> always a string
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "~"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)          # 0.5, 1e3 — units like "1.0Ti" stay strings
    except ValueError:
        return v


def _tokenize(text):
    toks = []
    for raw in text.split("\n"):
        if not raw.strip():
            continue
        stripped = raw.lstrip()
        if stripped.startswith("#"):
            continue
        ci = raw.find(" #")          # strip inline comments (no '#' in our values)
        if ci != -1:
            raw = raw[:ci]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        toks.append([indent, raw.strip()])
    return toks


class _Parser:
    def __init__(self, toks):
        self.t = toks
        self.i = 0

    def value(self, indent):
        if self.i >= len(self.t):
            return None
        if self.t[self.i][1].startswith("- "):
            return self.lst(indent)
        return self.mp(indent)

    def mp(self, indent):
        d = {}
        while self.i < len(self.t):
            ind, content = self.t[self.i]
            if ind != indent or content.startswith("- "):
                break
            key, _, val = content.partition(":")
            key, val = key.strip(), val.strip()
            if val.startswith("&"):              # anchor def -> value is nested block
                parts = val.split(None, 1)
                val = parts[1] if len(parts) > 1 else ""
            self.i += 1
            if val == "":
                if self.i < len(self.t) and self.t[self.i][0] > indent:
                    d[key] = self.value(self.t[self.i][0])
                else:
                    d[key] = None
            else:
                d[key] = None if val.startswith("*") else _scalar(val)
        return d

    def lst(self, indent):
        out = []
        while self.i < len(self.t):
            ind, content = self.t[self.i]
            if ind != indent or not content.startswith("-"):
                break
            rest = content[1:].strip()
            if rest == "":
                self.i += 1
                if self.i < len(self.t) and self.t[self.i][0] > indent:
                    out.append(self.value(self.t[self.i][0]))
                else:
                    out.append(None)
            elif ":" in rest:                    # mapping item
                self.t[self.i] = [indent + 2, rest]
                out.append(self.value(indent + 2))
            else:                                # scalar item
                out.append(_scalar(rest))
                self.i += 1
        return out


def _merge_local(reg, local):
    """Overlay private connection details from the gitignored machines.local.yaml
    onto the tracked registry, matched by node name. Keeps intranet IPs, real
    hostnames and ssh host/port/user out of the committed machines.yaml."""
    import os
    p = Path(os.environ.get("COWORK_LOCAL") or local)
    if not p.exists():
        return reg
    overlay = _Parser(_tokenize(p.read_text())).mp(0)
    by_name = {m.get("name"): m for m in machines(reg)}
    for node in (overlay.get("machines") or []):
        m = by_name.get(node.get("name"))
        if m is None:
            continue
        for k, v in node.items():
            if k == "name":
                continue
            if k == "ssh" and isinstance(v, dict) and isinstance(m.get("ssh"), dict):
                m["ssh"].update(v)
            else:
                m[k] = v
    return reg


def load_registry(path=REG, local=LOCAL):
    toks = _tokenize(Path(path).read_text())
    return _merge_local(_Parser(toks).mp(0), local)


def machines(reg):
    return reg.get("machines", []) or []


def machine_by_name(reg, name):
    for m in machines(reg):
        if m.get("name") == name:
            return m
    return None
