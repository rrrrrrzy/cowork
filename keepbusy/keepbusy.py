import argparse
import gc
import os
import signal
import subprocess
import sys
import time
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


DEFAULT_SCALE = 2000
DEFAULT_UTIL_THRESHOLD = 5.0
DEFAULT_IDLE_MINUTES = 10.0
DEFAULT_POLL_INTERVAL = 5.0

_interrupt_requested = False


def log(message: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def parse_gpu_list(value: str) -> List[int]:
    if value is None or value == "":
        return []
    try:
        gpus = [int(x.strip()) for x in value.split(",") if x.strip() != ""]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--gpus expects a comma-separated list of integers, got {value!r}"
        ) from exc
    if any(g < 0 for g in gpus):
        raise argparse.ArgumentTypeError("GPU indices must be non-negative")
    return gpus


def handle_signal(signum, _frame):
    global _interrupt_requested
    _interrupt_requested = True
    signal_name = signal.Signals(signum).name
    log(f">>> received {signal_name}")


class ComplexModel(nn.Module):
    def __init__(self, scale: int):
        super().__init__()
        self.layer1 = nn.Linear(10 * scale, 5 * scale)
        self.bn1 = nn.BatchNorm1d(5 * scale)
        self.layer2 = nn.Linear(5 * scale, 10 * scale)
        self.bn2 = nn.BatchNorm1d(10 * scale)
        self.layer3 = nn.Linear(10 * scale, 5 * scale)
        self.bn3 = nn.BatchNorm1d(5 * scale)
        self.layer4 = nn.Linear(5 * scale, 10 * scale)

    def forward(self, x):
        x = F.relu(self.bn1(self.layer1(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = F.relu(self.bn2(self.layer2(x)))
        x = F.relu(self.bn3(self.layer3(x)))
        x = self.layer4(x)
        return x


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor GPU utilization and keep GPUs busy only after sustained idleness."
    )
    parser.add_argument(
        "--util-threshold",
        type=float,
        default=DEFAULT_UTIL_THRESHOLD,
        help="Treat a GPU as idle when utilization is below this percentage.",
    )
    parser.add_argument(
        "--idle-minutes",
        type=float,
        default=DEFAULT_IDLE_MINUTES,
        help="Required continuous idle time before keepbusy starts.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between utilization checks.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=DEFAULT_SCALE,
        help="Model scale factor for the synthetic workload.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=6400,
        help="Synthetic batch size used during keepbusy.",
    )
    parser.add_argument(
        "--disable-auto-interrupt",
        action="store_true",
        help="Keep running even if another GPU compute process appears.",
    )
    parser.add_argument(
        "--gpus",
        type=parse_gpu_list,
        default=[],
        help="Comma-separated list of physical GPU indices to use (e.g. '0,1'). "
             "Default: all visible GPUs.",
    )
    return parser.parse_args()


def ensure_cuda():
    if not torch.cuda.is_available():
        log("CUDA is not available. Exiting...")
        sys.exit(1)


def _id_arg(gpu_ids: List[int]) -> List[str]:
    if not gpu_ids:
        return []
    return [f"--id={','.join(str(g) for g in gpu_ids)}"]


def run_nvidia_smi(query: str, gpu_ids: List[int]) -> List[str]:
    command = [
        "nvidia-smi",
        f"--query-gpu={query}",
        "--format=csv,noheader,nounits",
        *_id_arg(gpu_ids),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def get_gpu_utilizations(gpu_ids: List[int]) -> List[float]:
    rows = run_nvidia_smi("utilization.gpu", gpu_ids)
    return [float(row) for row in rows]


def get_compute_pids(gpu_ids: List[int]) -> List[int]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
        *_id_arg(gpu_ids),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return []

    pids = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or line == "[Not Supported]":
            continue
        pid = int(line)
        pids.append(pid)
    return sorted(set(pids))


def all_gpus_idle(utilizations: List[float], threshold: float) -> bool:
    return bool(utilizations) and all(util < threshold for util in utilizations)


def build_model(scale: int):
    model = ComplexModel(scale).cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    return model


def _empty_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    log(">>> released GPU memory back to the driver")


def keepbusy(args) -> str:
    global _interrupt_requested

    model = build_model(args.scale)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    data = torch.randn(args.batch_size, 10 * args.scale, device="cuda")
    target = torch.randint(
        0,
        10 * args.scale,
        (args.batch_size,),
        dtype=torch.long,
        device="cuda",
    )
    output = None
    loss = None

    log(f">>> keepbusy started on {torch.cuda.device_count()} GPU(s)")

    try:
        while True:
            if _interrupt_requested:
                _interrupt_requested = False
                log(">>> keepbusy interrupted manually, returning to monitor stage")
                return "monitor"

            if not args.disable_auto_interrupt:
                compute_pids = get_compute_pids(args.gpus)
                if len(compute_pids) >= 2:
                    log(
                        ">>> detected at least 2 GPU compute processes "
                        f"(pid(s): {', '.join(map(str, compute_pids))}), returning to monitor stage"
                    )
                    return "monitor"

            output = model(data)
            loss = F.cross_entropy(output, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    finally:
        del output, loss, data, target, optimizer, model
        _empty_cuda_cache()


def monitor(args):
    global _interrupt_requested

    required_idle_seconds = args.idle_minutes * 60.0
    idle_since = None

    log(
        ">>> monitoring GPU utilization "
        f"(threshold < {args.util_threshold:.1f}% for {args.idle_minutes:.1f} minute(s))"
    )

    while True:
        if _interrupt_requested:
            log(">>> interrupt received during monitor stage, exiting")
            return

        try:
            utilizations = get_gpu_utilizations(args.gpus)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            log(f"Failed to query GPU utilization via nvidia-smi: {exc}")
            return

        if not utilizations:
            log("No GPUs reported by nvidia-smi. Exiting...")
            return

        now = time.time()
        if all_gpus_idle(utilizations, args.util_threshold):
            if idle_since is None:
                idle_since = now
            idle_seconds = now - idle_since
            remaining = max(required_idle_seconds - idle_seconds, 0.0)
            log(
                f">>> idle GPUs detected {utilizations}; "
                f"keepbusy starts in {remaining:.0f}s if idleness continues"
            )
            if idle_seconds >= required_idle_seconds:
                keepbusy(args)
                idle_since = None
        else:
            if idle_since is not None:
                log(f">>> GPU activity resumed {utilizations}; idle timer reset")
            else:
                log(f">>> GPUs active {utilizations}")
            idle_since = None

        time.sleep(args.poll_interval)


def main():
    args = parse_args()
    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in args.gpus)
        log(f">>> restricting to physical GPU(s): {args.gpus}")
    ensure_cuda()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    monitor(args)


if __name__ == "__main__":
    main()
