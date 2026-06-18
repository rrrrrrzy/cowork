"""cowork — file-queue job scheduling across the shared-NFS dev cluster.

Modules:
  registry  — minimal YAML-subset parser for machines.yaml (no PyYAML).
  probe     — concurrent nvidia-smi probing (local + ssh, ControlMaster reuse).
  state     — paths + atomic file ops for the /share coordination board.
  agent     — the per-node daemon: claim jobs, run them, supervise keepbusy.
  cli       — the `cowork` command surface (submit/ps/logs/cancel/status/up/down).

Standard library only — runs from the shared miniconda python on every node.
"""
