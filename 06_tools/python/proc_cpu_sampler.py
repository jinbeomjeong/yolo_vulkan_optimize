#!/usr/bin/env python3
"""Sample CPU ticks of one target process, the whole system and labwc over a run. Standard library only.

Used by 06_tools/bench/demo_compare_bench.sh. Writes one CSV (at exit; nothing is written during the run):

  mono_ms,event,pid,pid_ticks,sys_busy_ticks,sys_total_ticks,labwc_ticks,sampler_ticks,clk_tck

  mono_ms         CLOCK_MONOTONIC in ms when the row was sampled (std::chrono::steady_clock is the same clock)
  event           start | tick | enable | disable | end
                  enable / disable = the target announced its CPU window on --ctl (perf_window() in
                  01_src/toolkit/tool_util.hpp). enable is written right before the target takes t0, the time base of
                  its --frame-log, so frame_log ms + enable mono_ms puts a frame on this file's time axis. The read
                  delay is this process's wake-up latency: compare with yvo_cpu_baseline's t0_monotonic_ms line.
  pid             target pid; empty until --pid-file appears
  pid_ticks       utime+stime of the target, all its threads (/proc/<pid>/stat); empty when not running
  sys_busy_ticks  /proc/stat "cpu" line: total minus idle and iowait (includes this sampler: see sampler_ticks)
  sys_total_ticks /proc/stat "cpu" line: sum of all fields up to steal
  labwc_ticks     utime+stime of the labwc process; empty when there is none
  sampler_ticks   utime+stime of this sampler. pid_ticks never contains it (a separate process); subtract it from
                  sys_busy_ticks for the system load without the sampler
  clk_tck         ticks per second

The sampler ends when the target exits, on SIGTERM/SIGINT, or after --max-s. Exit code 0; 2 on bad arguments or a CSV
that cannot be written; 3 when the target never appeared.
"""
import argparse
import os
import select
import signal
import sys
import time

COLUMNS = "mono_ms,event,pid,pid_ticks,sys_busy_ticks,sys_total_ticks,labwc_ticks,sampler_ticks,clk_tck"
CLK_TCK = os.sysconf("SC_CLK_TCK")


def now_ms():
    return time.clock_gettime(time.CLOCK_MONOTONIC) * 1000.0


def proc_ticks(pid):
    """utime+stime of pid, or None when it is gone or a zombie. comm may hold spaces: fields count from the last ')'."""
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read()
    except OSError:
        return None
    rest = data[data.rfind(b")") + 2:].split()
    if rest[0] == b"Z":
        return None
    return int(rest[11]) + int(rest[12])   # fields 14 and 15 of proc(5)


def system_ticks():
    """(busy, total) from the aggregate "cpu" line. guest is already inside user, so only the first 8 fields count."""
    with open("/proc/stat", "rb") as f:
        fields = [int(x) for x in f.readline().split()[1:9]]
    total = sum(fields)
    return total - fields[3] - fields[4], total


def find_labwc():
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/comm", "rb") as f:
                if f.read().strip() == b"labwc":
                    return int(name)
        except OSError:
            continue
    return None


def read_pid_file(path):
    try:
        with open(path) as f:
            text = f.read().strip()
    except OSError:
        return None
    return int(text) if text.isdigit() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = ap.add_mutually_exclusive_group(required=True)
    target.add_argument("--pid", type=int, help="target pid")
    target.add_argument("--pid-file", help="file that will hold the target pid (polled)")
    ap.add_argument("--ctl", help="FIFO the target writes enable/disable lines to (YVO_PERF_CTL)")
    ap.add_argument("--out", required=True, help="output CSV")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between tick rows (default 0.5)")
    ap.add_argument("--pid-timeout", type=float, default=30.0, help="seconds to wait for --pid-file (default 30)")
    ap.add_argument("--max-s", type=float, default=3600.0, help="longest run in seconds (default 3600)")
    args = ap.parse_args()
    if args.interval <= 0:
        ap.error("--interval must be > 0")

    stop = {"flag": False}
    def on_signal(_sig, _frame):
        stop["flag"] = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    ctl_fd = None
    if args.ctl:
        # O_RDWR: never blocks on open and never sees EOF between writers.
        ctl_fd = os.open(args.ctl, os.O_RDWR | os.O_NONBLOCK | os.O_CLOEXEC)
    poller = select.poll()
    if ctl_fd is not None:
        poller.register(ctl_fd, select.POLLIN)

    labwc = find_labwc()
    self_pid = os.getpid()
    rows = []                 # kept in memory: no file I/O while the target runs
    pid = args.pid
    state = {"seen": False}

    def sample(event):
        if pid is not None:
            ticks = proc_ticks(pid)
            state["seen"] = state["seen"] or ticks is not None
        else:
            ticks = None
        busy, total = system_ticks()
        lw = proc_ticks(labwc) if labwc is not None else None
        rows.append((now_ms(), event, pid, ticks, busy, total, lw, proc_ticks(self_pid), CLK_TCK))
        return ticks

    sample("start")
    t_begin = time.monotonic()
    next_tick = t_begin + args.interval
    pending = b""
    while not stop["flag"]:
        now = time.monotonic()
        if now - t_begin > args.max_s:
            break
        if pid is None:
            pid = read_pid_file(args.pid_file)
            if pid is None and now - t_begin > args.pid_timeout:
                break
        timeout_ms = max(0, (next_tick - now) * 1000.0)
        if pid is None:
            timeout_ms = min(timeout_ms, 20.0)
        try:
            events = poller.poll(timeout_ms)
        except InterruptedError:
            continue
        if events:
            try:
                pending += os.read(ctl_fd, 256)
            except BlockingIOError:
                pass
            while b"\n" in pending:
                line, pending = pending.split(b"\n", 1)
                word = line.strip().decode("ascii", "replace")
                if word in ("enable", "disable"):
                    sample(word)
        if time.monotonic() >= next_tick:
            next_tick += args.interval
            if sample("tick") is None and state["seen"]:
                break
    sample("end")

    try:
        with open(args.out, "w") as f:
            f.write(COLUMNS + "\n")
            for r in rows:
                f.write(f"{r[0]:.3f},{r[1]}," + ",".join("" if v is None else str(v) for v in r[2:]) + "\n")
    except OSError as e:
        print(f"proc_cpu_sampler: cannot write {args.out}: {e}", file=sys.stderr)
        return 2
    return 0 if state["seen"] else 3


if __name__ == "__main__":
    sys.exit(main())
