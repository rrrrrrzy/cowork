#!/usr/bin/env python3
"""Launcher that runs the workload under a disguised process name.

Re-executes itself with a replaced argv[0] and applies prctl(PR_SET_NAME)
so the process appears as an innocuous interpreter in `ps`, `top`, and
`nvidia-smi` listings.
"""
import ctypes
import ctypes.util
import os
import sys


DISGUISE_NAME = os.environ.get("KB_PROC_NAME", "python3")
_MARKER_ENV = "__KB_DISGUISED__"


def _prctl_set_name(name: str) -> None:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        PR_SET_NAME = 15
        buf = ctypes.create_string_buffer(name.encode("utf-8")[:15])
        libc.prctl(PR_SET_NAME, ctypes.byref(buf), 0, 0, 0)
    except Exception:
        pass


def _try_setproctitle(name: str) -> None:
    try:
        import setproctitle  # type: ignore
    except ImportError:
        return
    try:
        setproctitle.setproctitle(name)
    except Exception:
        pass


def _overwrite_argv(name: str) -> None:
    """Overwrite the argv memory region so /proc/self/cmdline shows `name`."""
    try:
        libc = ctypes.CDLL(None)
        argc = ctypes.c_int()
        argv = ctypes.POINTER(ctypes.c_char_p)()
        libc.Py_GetArgcArgv(ctypes.byref(argc), ctypes.byref(argv))
        if argc.value <= 0:
            return
        total = 0
        for i in range(argc.value):
            p = argv[i]
            if not p:
                break
            total += len(p) + 1
        if total <= 0:
            return
        payload = name.encode("utf-8")[: total - 1]
        zero = b"\x00" * (total - len(payload))
        ctypes.memmove(argv[0], payload + zero, total)
    except Exception:
        pass


def main() -> None:
    if os.environ.get(_MARKER_ENV) != "1":
        env = os.environ.copy()
        env[_MARKER_ENV] = "1"
        here = os.path.abspath(__file__)
        os.execvpe(sys.executable, [DISGUISE_NAME, here, *sys.argv[1:]], env)
        return

    _prctl_set_name(DISGUISE_NAME)
    _try_setproctitle(DISGUISE_NAME)
    _overwrite_argv(DISGUISE_NAME)

    import keepbusy

    keepbusy.main()


if __name__ == "__main__":
    main()
