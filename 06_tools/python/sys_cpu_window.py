#!/usr/bin/env python3
"""System-wide CPU over exactly the CPU window of a measured tool (yvo_pipeline and friends).

The tools mark their CPU window through perf's control-FIFO protocol (01_src/toolkit/tool_util.hpp, perf_window()):
with YVO_PERF_CTL=<ctl> the tool opens <ctl> O_WRONLY and writes "enable\\n" at the window start and "disable\\n" at
the end; with YVO_PERF_ACK=<ack> it opens <ack> O_RDONLY and waits (poll, 1000 ms) for "ack\\n" after each command.
This script stands in for perf on that protocol: it takes a /proc snapshot on "enable", another on "disable", acks
each after its snapshot, and writes the difference as JSON. Runs on system python3 (standard library only).

Window mode (the bench waits for --ready, then starts the tool with YVO_PERF_CTL/YVO_PERF_ACK set):

    python3 06_tools/python/sys_cpu_window.py --ctl <fifo> --ack <fifo> --out <path.json> [--ready <file>] [--timeout 600]

Baseline mode (no tool; snapshot, sleep, snapshot):

    python3 06_tools/python/sys_cpu_window.py --duration <sec> --out <path.json>

Common options: --frames N (adds *_ms_per_frame), --app-comm yvo_pipeline (comm of the measured process),
--csv-fields (print one summary CSV line, no header, to stdout:
win_ms,sys_busy_ms,sched_run_ms,app_ms,labwc_ms,kthread_ms,other_ms,softirq_ms,irq_count).

FIFO handling:
  - Missing FIFOs are created (mkfifo) and left in place on exit.
  - <ctl> is opened O_RDONLY | O_NONBLOCK and <ack> O_RDWR | O_NONBLOCK before --ready is written, so the tool's
    open(ctl, O_WRONLY) and open(ack, O_RDONLY) return at once (same effect as perf's O_RDWR | O_NONBLOCK). <ctl> is
    read-only here on purpose: holding no write end lets this script see EOF (POLLHUP) when the tool exits.

Exit codes: 0 = full window recorded, 3 = partial (no enable, or no disable, before timeout/EOF/signal; the JSON
"status" says which), 2 = usage or FIFO error.

Notes on the numbers:
  - sys_busy_ms is user+nice+system+irq+softirq+steal from the aggregate "cpu" line of /proc/stat (ticks of
    1/clk_tck s). Kernels without IRQ time accounting report irq (and possibly softirq) as 0.
  - sched_run_ms is the sum over CPUs of the 7th value of the cpuN lines of /proc/schedstat (ns spent running tasks).
  - Per-process ms are utime+stime ticks of /proc/<pid>/stat (includes exited threads, not reaped children), so the
    resolution is one tick (10 ms at clk_tck 100) per process.
  - A process that starts and exits inside the window is in neither snapshot, so it is in no class. Its CPU is
    visible as unattributed_ms (sys_busy_ms minus the classes) and, once reaped, in reaped_children_ms.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import os
import select
import signal
import stat
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

STAT_COLS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice")
BUSY_COLS = ("user", "nice", "system", "irq", "softirq", "steal")
SCHEDSTAT_VERSION = 17
KTHREAD_GROUPS = (("kworker", "kworker*"), ("ksoftirqd", "ksoftirqd*"), ("irq_thread", "irq/*"),
                  ("v3d", "v3d*"), ("hailo", "hailo*"))
TOP_N_PROCS = 10
TOP_N_IRQS = 15
MAX_LISTED = 50
CSV_FIELDS = ("win_ms", "sys_busy_ms", "sched_run_ms", "app_ms", "labwc_ms", "kthread_ms", "other_ms",
              "softirq_ms", "irq_count")

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PARTIAL = 3


@dataclass
class Proc:
    pid: int
    comm: str
    state: str
    ppid: int
    ticks: int  # utime + stime
    starttime: int
    child_ticks: int  # cutime + cstime (reaped children)


@dataclass
class Snapshot:
    t_ns: int  # CLOCK_MONOTONIC right before /proc/stat is read
    took_ms: float
    stat_lines: dict[str, list[int]]  # "cpu", "cpu0", ..., "ctxt", "softirq", ...
    schedstat_version: int | None
    sched_run_ns: dict[str, int]
    interrupts: dict[str, tuple[str, int]]  # label -> (description, sum over CPUs)
    procs: dict[int, Proc] = field(default_factory=dict)


class Stop(Exception):
    """Raised by the SIGTERM/SIGINT handler to end the wait loop."""


def _read(path: str) -> bytes:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        chunks = []
        while True:
            b = os.read(fd, 65536)
            if not b:
                break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)


def _parse_stat(data: bytes) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for line in data.decode("ascii", "replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = [int(x) for x in parts[1:]]
        except ValueError:
            continue
    return out


def _parse_schedstat(data: bytes) -> tuple[int | None, dict[str, int]]:
    version: int | None = None
    run: dict[str, int] = {}
    for line in data.decode("ascii", "replace").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "version" and len(parts) > 1:
            version = int(parts[1])
        elif parts[0].startswith("cpu") and len(parts) > 7:
            run[parts[0]] = int(parts[7])  # 7th value after the cpuN token: task run time [ns]
    return version, run


def _parse_interrupts(data: bytes) -> dict[str, tuple[str, int]]:
    lines = data.decode("ascii", "replace").splitlines()
    if not lines:
        return {}
    ncols = len(lines[0].split())
    out: dict[str, tuple[str, int]] = {}
    for line in lines[1:]:
        label, sep, rest = line.partition(":")
        if not sep:
            continue
        toks = rest.split()
        total = 0
        n = 0
        while n < len(toks) and n < ncols and toks[n].isdigit():
            total += int(toks[n])
            n += 1
        if n == 0:
            continue
        out[label.strip()] = (" ".join(toks[n:]), total)
    return out


def _read_procs() -> dict[int, Proc]:
    procs: dict[int, Proc] = {}
    with os.scandir("/proc") as it:
        for entry in it:
            name = entry.name
            if not name.isdigit():
                continue
            try:
                data = _read(f"/proc/{name}/stat")
            except OSError:  # exited between scandir and open/read
                continue
            lp = data.find(b"(")
            rp = data.rfind(b")")
            if lp < 0 or rp < lp:
                continue
            rest = data[rp + 2:].split()
            if len(rest) < 20:
                continue
            # rest[0] is field 3 (state); field k is rest[k - 3].
            procs[int(name)] = Proc(
                pid=int(name),
                comm=data[lp + 1:rp].decode("utf-8", "replace"),
                state=rest[0].decode("ascii", "replace"),
                ppid=int(rest[1]),
                ticks=int(rest[11]) + int(rest[12]),
                starttime=int(rest[19]),
                child_ticks=int(rest[13]) + int(rest[14]),
            )
    return procs


def take_snapshot() -> Snapshot:
    t0 = time.perf_counter()
    t_ns = time.monotonic_ns()
    stat_lines = _parse_stat(_read("/proc/stat"))
    version, run = _parse_schedstat(_read("/proc/schedstat"))
    irqs = _parse_interrupts(_read("/proc/interrupts"))
    procs = _read_procs()
    took_ms = (time.perf_counter() - t0) * 1e3
    return Snapshot(t_ns, took_ms, stat_lines, version, run, irqs, procs)


def _kthread_group(comm: str) -> str:
    for name, pat in KTHREAD_GROUPS:
        if fnmatch.fnmatchcase(comm, pat):
            return name
    return "other"


def _top(items: list[dict], n: int) -> list[dict]:
    """The n largest entries by ms; entries with 0 ms are dropped."""
    return sorted((d for d in items if d["ms"] > 0), key=lambda d: d["ms"], reverse=True)[:n]


def _round(x: float) -> float:
    return round(x, 3)


def diff_snapshots(a: Snapshot, b: Snapshot, clk_tck: int, app_comm: str) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    tick_ms = 1e3 / clk_tck
    res: dict = {"window_ms": _round((b.t_ns - a.t_ns) / 1e6)}

    # /proc/stat
    ca, cb = a.stat_lines.get("cpu", []), b.stat_lines.get("cpu", [])
    cols = {}
    for i, name in enumerate(STAT_COLS):
        if i < len(ca) and i < len(cb):
            cols[name] = (cb[i] - ca[i]) * tick_ms
    res["sys_busy_ms"] = _round(sum(cols.get(c, 0.0) for c in BUSY_COLS))
    for name, v in cols.items():
        res[f"{name}_ms"] = _round(v)
    per_cpu = []
    k = 0
    while f"cpu{k}" in a.stat_lines and f"cpu{k}" in b.stat_lines:
        la, lb = a.stat_lines[f"cpu{k}"], b.stat_lines[f"cpu{k}"]
        busy = sum(lb[i] - la[i] for i, c in enumerate(STAT_COLS) if c in BUSY_COLS and i < len(la) and i < len(lb))
        per_cpu.append(_round(busy * tick_ms))
        k += 1
    res["per_cpu_busy_ms"] = per_cpu
    if "ctxt" in a.stat_lines and "ctxt" in b.stat_lines:
        res["ctxt_switches"] = b.stat_lines["ctxt"][0] - a.stat_lines["ctxt"][0]
    sa, sb = a.stat_lines.get("softirq", []), b.stat_lines.get("softirq", [])
    if sa and sb:
        names = ("total", "hi", "timer", "net_tx", "net_rx", "block", "irq_poll", "tasklet", "sched", "hrtimer", "rcu")
        res["softirq_counts"] = {names[i] if i < len(names) else f"vec{i}": sb[i] - sa[i]
                                 for i in range(min(len(sa), len(sb)))}

    # /proc/schedstat
    if a.schedstat_version != SCHEDSTAT_VERSION or b.schedstat_version != SCHEDSTAT_VERSION:
        warnings.append(f"schedstat version {a.schedstat_version}/{b.schedstat_version} != {SCHEDSTAT_VERSION}")
    res["schedstat_version"] = b.schedstat_version
    common = sorted(set(a.sched_run_ns) & set(b.sched_run_ns))
    res["sched_run_ms"] = _round(sum(b.sched_run_ns[c] - a.sched_run_ns[c] for c in common) / 1e6) if common else None

    # Processes
    sums = {"app": 0.0, "labwc": 0.0, "kthread": 0.0, "other": 0.0}
    kgroups = {name: 0.0 for name, _ in KTHREAD_GROUPS} | {"other": 0.0}
    kthread_items: list[dict] = []
    other_items: list[dict] = []
    app_items: list[dict] = []
    new_procs: list[dict] = []
    lost_procs: list[dict] = []
    self_ms = 0.0
    reaped_ticks = 0
    me = os.getpid()
    for pid, pb in b.procs.items():
        pa = a.procs.get(pid)
        if pa is not None and pa.starttime == pb.starttime:
            ms = (pb.ticks - pa.ticks) * tick_ms
            reaped_ticks += pb.child_ticks - pa.child_ticks
        else:
            ms = pb.ticks * tick_ms  # born after A (or pid reused): all of its CPU is inside the window
            new_procs.append({"pid": pid, "comm": pb.comm, "ppid": pb.ppid, "ms": _round(ms)})
        item = {"pid": pid, "comm": pb.comm, "ms": _round(ms)}
        if pid == me:
            self_ms = ms
        if pb.comm == app_comm:
            sums["app"] += ms
            app_items.append(item)
        elif pb.comm == "labwc":
            sums["labwc"] += ms
        elif pid == 2 or pb.ppid == 2:
            sums["kthread"] += ms
            kgroups[_kthread_group(pb.comm)] += ms
            kthread_items.append(item)
        else:
            sums["other"] += ms
            other_items.append(item)
    for pid, pa in a.procs.items():
        pb = b.procs.get(pid)
        if pb is None or pb.starttime != pa.starttime:
            lost_procs.append({"pid": pid, "comm": pa.comm, "ppid": pa.ppid})
    for name, v in sums.items():
        res[f"{name}_ms"] = _round(v)
    # CPU of processes that started and exited inside the window is in neither snapshot. It shows up here:
    # unattributed = /proc/stat busy minus the four classes; reaped_children = growth of cutime+cstime of the
    # processes alive at A and B (approximate: a child alive at A and reaped later brings its pre-A CPU too).
    res["unattributed_ms"] = _round(res["sys_busy_ms"] - sum(sums.values()))
    res["reaped_children_ms"] = _round(reaped_ticks * tick_ms)
    res["procs"] = {
        "app_comm": app_comm,
        "app_pids": sorted(app_items, key=lambda d: d["pid"]),
        "kthread_groups": {f"{k}_ms": _round(v) for k, v in kgroups.items()},
        "kthread_top": _top(kthread_items, TOP_N_PROCS),
        "other_top": _top(other_items, TOP_N_PROCS),
        "sampler_self_ms": _round(self_ms),
        "new_proc_count": len(new_procs),
        "new_procs": sorted(new_procs, key=lambda d: d["ms"], reverse=True)[:MAX_LISTED],
        "lost_proc_count": len(lost_procs),
        "lost_procs": lost_procs[:MAX_LISTED],
        "count_a": len(a.procs),
        "count_b": len(b.procs),
    }
    if not app_items:
        warnings.append(f"no process with comm '{app_comm}' in snapshot B")

    # /proc/interrupts
    irq_items = []
    total = 0
    for label, (desc, nb) in b.interrupts.items():
        na = a.interrupts.get(label, (desc, 0))[1]
        d = nb - na
        total += d
        if d > 0:
            irq_items.append({"irq": label, "desc": desc, "count": d})
    irq_items.sort(key=lambda d: d["count"], reverse=True)
    res["interrupts"] = {"total": total, "top": irq_items[:TOP_N_IRQS]}
    res["irq_count"] = total
    return res, warnings


def add_per_frame(d: dict, frames: int) -> None:
    """Add <key>_per_frame next to every numeric *_ms key of d and of its nested dicts (lists are left alone)."""
    for key in list(d):
        v = d[key]
        if isinstance(v, dict):
            add_per_frame(v, frames)
        elif key.endswith("_ms") and isinstance(v, (int, float)) and not isinstance(v, bool):
            d[f"{key}_per_frame"] = round(v / frames, 4)


def csv_line(res: dict) -> str:
    vals = []
    for f in CSV_FIELDS:
        v = res.get("window_ms" if f == "win_ms" else f)
        vals.append("nan" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{float(v):.1f}")
    return ",".join(vals)


def ensure_fifo(path: Path) -> None:
    try:
        st = path.stat()
    except FileNotFoundError:
        os.mkfifo(path, 0o600)
        return
    if not stat.S_ISFIFO(st.st_mode):
        raise OSError(f"{path} exists and is not a FIFO")


class CtlReader:
    """Line reader over the non-blocking read end of the ctl FIFO. EOF = the writer was there and is gone."""

    def __init__(self, fd: int) -> None:
        self.fd = fd
        self.buf = b""
        self.eof = False
        self.poller = select.poll()
        self.poller.register(fd, select.POLLIN)

    def next_line(self, deadline: float) -> str | None:
        """Next command line, or None on timeout/EOF (check self.eof)."""
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                line, self.buf = self.buf[:nl], self.buf[nl + 1:]
                return line.decode("ascii", "replace").strip()
            if self.eof:
                return None
            left_ms = (deadline - time.monotonic()) * 1e3
            if left_ms <= 0:
                return None
            events = self.poller.poll(min(left_ms, 1000.0))
            if not events:
                continue
            try:
                data = os.read(self.fd, 4096)
            except BlockingIOError:
                continue
            if data:
                self.buf += data
            else:  # POLLHUP after the writer closed (Linux reports no HUP before a writer ever opened)
                self.eof = True


def write_result(out: Path, res: dict, csv: bool) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
        f.write("\n")
    if csv:
        print(csv_line(res), flush=True)


def build_result(mode: str, a: Snapshot | None, b: Snapshot | None, args: argparse.Namespace,
                 status: str) -> dict:
    clk_tck = os.sysconf("SC_CLK_TCK")
    res: dict = {
        "tool": "sys_cpu_window",
        "mode": mode,
        "status": status,
        "clk_tck": clk_tck,
        "ncpu": os.cpu_count(),
        "app_comm": args.app_comm,
        "frames": args.frames,
        "window_ms": None,
        "snapshot_ms": {"a": _round(a.took_ms) if a else None, "b": _round(b.took_ms) if b else None},
    }
    if a is not None and b is not None:
        diff, warnings = diff_snapshots(a, b, clk_tck, args.app_comm)
        res.update(diff)
        if warnings:
            res["status"] = status + "; warning: " + "; ".join(warnings)
        if args.frames:
            add_per_frame(res, args.frames)
    return res


def run_window(args: argparse.Namespace) -> int:
    ctl, ack = Path(args.ctl), Path(args.ack)
    try:
        ensure_fifo(ctl)
        ensure_fifo(ack)
        ctl_fd = os.open(ctl, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
        ack_fd = os.open(ack, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    except OSError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    if args.ready:
        Path(args.ready).write_text(f"{os.getpid()}\n", encoding="utf-8")

    reader = CtlReader(ctl_fd)
    deadline = time.monotonic() + args.timeout
    a: Snapshot | None = None
    b: Snapshot | None = None
    status = "ok"
    try:
        while True:
            line = reader.next_line(deadline)
            if line is None:
                why = "eof" if reader.eof else "timeout"
                status = f"partial: {'no disable' if a else 'no enable'} before {why}"
                break
            if line == "enable" and a is None:
                a = take_snapshot()
                os.write(ack_fd, b"ack\n")
            elif line == "disable" and a is not None:
                b = take_snapshot()
                os.write(ack_fd, b"ack\n")
                break
            else:
                os.write(ack_fd, b"ack\n")  # unexpected/duplicate command: ack so the tool does not stall
                print(f"warning: ignored control command '{line}'", file=sys.stderr)
    except Stop:
        status = f"partial: {'no disable' if a else 'no enable'} before signal"
    finally:
        os.close(ctl_fd)
        os.close(ack_fd)

    if a is not None and b is None:
        b = take_snapshot()  # keep the partial window (enable .. now) for diagnosis
    write_result(Path(args.out), build_result("window", a, b, args, status), args.csv_fields)
    return EXIT_OK if status == "ok" else EXIT_PARTIAL


def run_duration(args: argparse.Namespace) -> int:
    status = "ok"
    a = take_snapshot()
    try:
        time.sleep(args.duration)
    except Stop:
        status = "partial: signal during sleep"
    b = take_snapshot()
    write_result(Path(args.out), build_result("duration", a, b, args, status), args.csv_fields)
    return EXIT_OK if status == "ok" else EXIT_PARTIAL


def _on_signal(signum: int, _frame: object) -> None:
    raise Stop(signal.Signals(signum).name)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="System-wide CPU over a tool's perf_window() CPU window (or a fixed "
                                            "duration), from /proc snapshots.")
    p.add_argument("--ctl", help="control FIFO the tool writes enable/disable to (YVO_PERF_CTL)")
    p.add_argument("--ack", help="ack FIFO the tool waits on (YVO_PERF_ACK)")
    p.add_argument("--ready", help="file created once both FIFOs are open (start the tool after it appears)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for enable+disable (default 600)")
    p.add_argument("--duration", type=float, help="baseline mode: seconds between the two snapshots")
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--frames", type=int, help="frame count of the window; adds *_ms_per_frame")
    p.add_argument("--app-comm", default="yvo_pipeline", help="comm of the measured process (default yvo_pipeline)")
    p.add_argument("--csv-fields", action="store_true",
                   help="print one CSV line (no header) to stdout: " + ",".join(CSV_FIELDS))
    args = p.parse_args(argv)

    if args.frames is not None and args.frames <= 0:
        p.error("--frames must be positive")
    if args.duration is not None:
        if args.ctl or args.ack:
            p.error("--duration cannot be combined with --ctl/--ack")
        if args.duration < 0:
            p.error("--duration must be >= 0")
    elif not (args.ctl and args.ack):
        p.error("either --ctl and --ack (window mode) or --duration (baseline mode) is required")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    return run_duration(args) if args.duration is not None else run_window(args)


if __name__ == "__main__":
    sys.exit(main())
