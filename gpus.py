#!/share/ziyangrao/miniconda3/bin/python3
"""cowork — cluster GPU monitor.

Reads machines.yaml, connects to every node concurrently (ssh for remotes,
local exec for the node we're running on), queries nvidia-smi, and prints
free/busy GPUs per node plus the tasks on each GPU (nvitop-style).

The registry parser and the nvidia-smi probe now live in the shared `cw` package
(cw.registry / cw.probe) so the scheduler, the per-node agent and this monitor all
use one implementation. Dependency-free: standard library only.

    ./gpus.py                 # DEFAULT: live GPU table (mem/util/free), refresh every 1s
    ./gpus.py --procs         # also list per-GPU tasks (pid/user/mem/runtime/cmd)
    ./gpus.py --once          # render once and exit
    ./gpus.py --watch 5       # refresh every 5s instead of 1
    ./gpus.py --json          # machine-readable, one-shot (proc_list per GPU)
    ./gpus.py --dump          # just show the parsed registry and exit
    ./gpus.py --mem-thresh 4096   # MiB; a GPU with <= this used and no procs is "free"
"""
import argparse
import atexit
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cw import probe                       # noqa: E402
from cw.registry import REG, load_registry  # noqa: E402


def render(results, show_procs=False):
    term_w = shutil.get_terminal_size((140, 40)).columns
    lines = []
    total_free = total_gpu = 0
    for r in results:
        if not r["ok"]:
            lines.append(f"\033[1m{r['name']:<12}\033[0m  \033[31mUNREACHABLE\033[0m  {r['error']}")
            lines.append("")
            continue
        n_free = sum(g["free"] for g in r["gpus"])
        n = len(r["gpus"])
        total_free += n_free
        total_gpu += n
        lines.append(f"\033[1m{r['name']:<12}\033[0m  {r['hostname']}   free {n_free}/{n}")
        all_procs = [p for g in r["gpus"] for p in g["proc_list"]]
        if show_procs and all_procs and not any(p["resolved"] for p in all_procs):
            lines.append("  \033[90m(GPU tasks run in other PID namespaces / containers — "
                         "only pid + mem visible here, not the command)\033[0m")
        lines.append(f"  {'gpu':>3}  {'mem(used/total MiB)':>22}  {'util%':>5}  {'procs':>5}  state")
        for g in r["gpus"]:
            stt = "\033[32mfree\033[0m" if g["free"] else "\033[33mbusy\033[0m"
            mem = f"{g['mem_used']:>7}/{g['mem_total']:<7}"
            lines.append(f"  {g['index']:>3}  {mem:>22}  {g['util']:>5}  {g['procs']:>5}  {stt}")
            if show_procs:
                for p in g["proc_list"]:
                    vis = (f"        gpu{g['index']} pid {p['pid']:>7}  "
                           f"{p['user']:<10} {p['gpu_mem']:>6}MiB  {p['runtime']:>12}  ")
                    if p["cmd"]:
                        cmd = p["cmd"]
                    elif not p["resolved"]:
                        cmd = "(command unavailable — task in another container)"
                    else:
                        cmd = "?"
                    avail = max(20, term_w - len(vis))
                    if len(cmd) > avail:
                        cmd = cmd[:avail - 1] + "…"
                    lines.append("\033[90m" + vis + cmd + "\033[0m")
        lines.append("")
    lines.append(f"\033[1mCLUSTER: {total_free}/{total_gpu} GPUs free\033[0m")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="cowork cluster GPU monitor")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--procs", "--tasks", action="store_true", dest="procs",
                    help="also list per-GPU tasks (pid/user/mem/runtime/command), nvitop-style")
    ap.add_argument("--dump", action="store_true", help="print parsed registry and exit")
    ap.add_argument("--watch", type=float, metavar="SEC", default=None,
                    help="refresh every SEC seconds (default: 1; 0 or --once = single shot)")
    ap.add_argument("--once", action="store_true",
                    help="render once and exit (disable the default live watch)")
    ap.add_argument("--timeout", type=float, default=10, help="ssh connect timeout (s)")
    ap.add_argument("--mem-thresh", type=int, default=2048,
                    help="MiB; GPU with <= this used and no procs counts as free")
    ap.add_argument("--no-reuse", dest="reuse", action="store_false",
                    help="disable SSH connection reuse (ControlMaster); fresh connect per probe")
    ap.add_argument("--registry", default=str(REG), help="path to machines.yaml")
    args = ap.parse_args()

    reg = load_registry(args.registry)
    machines = reg.get("machines", []) or []

    if args.dump:
        print(json.dumps(reg, indent=2))
        return

    if args.reuse and probe.has_remotes(machines):
        probe.ensure_cm_dir()
        atexit.register(probe.close_masters, machines)

    def cycle():
        with cf.ThreadPoolExecutor(max_workers=max(1, len(machines))) as ex:
            futs = [ex.submit(probe.probe, m, args.timeout, args.mem_thresh, args.reuse)
                    for m in machines]
            results = [f.result() for f in futs]
        order = {m.get("name"): i for i, m in enumerate(machines)}
        results.sort(key=lambda r: order.get(r["name"], 1e9))
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print(render(results, show_procs=args.procs))

    if args.once or args.watch == 0:
        interval = None
    elif args.watch is not None:
        interval = args.watch
    elif args.json:
        interval = None
    else:
        interval = 1.0

    if interval:
        try:
            while True:
                sys.stdout.write("\033[2J\033[H")
                print(f"cowork GPU monitor — every {interval:g}s — Ctrl-C to stop\n")
                cycle()
                time.sleep(interval)
        except KeyboardInterrupt:
            pass
    else:
        cycle()


if __name__ == "__main__":
    main()
