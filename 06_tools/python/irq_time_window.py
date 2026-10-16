#!/usr/bin/env python3
"""Hardirq/softirq time over a measured tool's CPU window, from irq tracepoints (perf script text).

Input: the text of

    perf record -a -e irq:irq_handler_entry,irq:irq_handler_exit,irq:softirq_entry,irq:softirq_exit ...
    perf script -F cpu,pid,tid,comm,time,event,trace

one event per line, e.g. (kernel 6.18 TP_printk formats "irq=%d name=%s", "irq=%d ret=%s", "vec=%u [action=%s]"):

     swapper     0/0     [002]  1234.567890: irq:irq_handler_entry: irq=45 name=v3d_core0
     swapper     0/0     [002]  1234.567893: irq:irq_handler_exit: irq=45 ret=handled
    yvo_pipeline 1234/1240 [001] 1234.5679: irq:softirq_entry: vec=1 [action=TIMER]

Parsing is anchored on "<sec>.<frac>: [irq:]<event>:" and "[<cpu>]"; columns are not assumed to be fixed width.
The comm is the text before the pid/tid token (it may contain spaces). Lines without a "[cpu]" token are skipped
(counted in parse.bad_lines). Runs on system python3 (standard library only).

Definitions:
  - Per CPU, entries and exits are paired on a stack: a hardirq by irq number, a softirq by vec. A frame that
    nests inside another (a hardirq interrupting a softirq, or the reverse) is subtracted from its parent, so
    every duration below is SELF time and hardirq + softirq totals never count the same nanosecond twice
    (incl_ms keeps the inclusive sum). An exit with no open entry (window start) and an entry with no exit
    (window end, or a lost exit found when a later exit/entry skips over it) are counted in "unmatched" and
    contribute no time.
  - The interrupted task is the pid/comm of the entry line. pid 0 (swapper, the idle task) = "idle" share,
    anything else = "busy" share.
  - window_s = last event time - first event time (all four events, all CPUs).
  - uncounted_ms = hardirq.idle_ms + softirq.idle_ms; uncounted_ms_per_frame = uncounted_ms / --frames.

Why the idle share is missing from process CPU accounting on this kernel (6.18 rpt, TICK_CPU_ACCOUNTING=y,
IRQ_TIME_ACCOUNTING not set, NO_HZ_IDLE=y): without IRQ time accounting, interrupt time is not tracked on its
own. While a task runs, the tick and the scheduler clock simply bill the interrupt to that task (its stime and
sum_exec_runtime), so the busy share is already inside some process's CPU. On an idle CPU the tick is stopped
and the whole sleep, including the hardirq/softirq work done in it without a task switch, is booked as idle time
of the idle task (pid 0), which no /proc/<pid> represents and /proc/schedstat does not count as run time. The
/proc/stat irq/softirq columns only see ticks that happen to land inside a handler. So the idle share is the
interrupt work that no CPU counter of sys_cpu_window.py includes; these tracepoints measure it directly.

Usage:

    /usr/bin/python3 06_tools/python/irq_time_window.py --script <perf_script.txt> [--frames N] --out <irq.json>
    /usr/bin/python3 06_tools/python/irq_time_window.py --csv-header

stdout is one CSV line (no header) with the columns of --csv-header:
irq_window_s,hardirq_count,hardirq_ms,hardirq_idle_ms,softirq_count,softirq_ms,softirq_idle_ms,unmatched,
uncounted_ms_per_frame. With 0 paired events (or an unreadable input) every column is nan; without --frames the
last column is nan.

The script never exits on an exception: what could be computed is written with an "error" field.
Exit codes: 0 = parsed, at least one paired event; 3 = JSON written but no paired event or "error" set;
2 = usage error or the --out file could not be written (the CSV line is still printed).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

CSV_FIELDS = ("irq_window_s", "hardirq_count", "hardirq_ms", "hardirq_idle_ms", "softirq_count", "softirq_ms",
              "softirq_idle_ms", "unmatched", "uncounted_ms_per_frame")
# include/linux/interrupt.h order, also the TP_printk symbol table of include/trace/events/irq.h.
SOFTIRQ_NAMES = ("HI", "TIMER", "NET_TX", "NET_RX", "BLOCK", "IRQ_POLL", "TASKLET", "SCHED", "HRTIMER", "RCU")
HARD = "hardirq"
SOFT = "softirq"

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_PARTIAL = 3

# "<prefix> <sec>.<frac>: [<subsys>:]<event>: <trace>" -- perf right-aligns the event name, so allow any spacing.
LINE_RE = re.compile(r"^(?P<pre>.*?)(?<![\w.])(?P<sec>\d+)\.(?P<frac>\d+):\s*(?:[\w.-]+:)?"
                     r"(?P<ev>irq_handler_entry|irq_handler_exit|softirq_entry|softirq_exit):\s*(?P<trace>.*?)\s*$")
# Fast path for perf's default layout "<comm> <pid>/<tid> [<cpu>] <time>:"; anything else goes through LINE_RE.
FAST_RE = re.compile(r"^\s*(?P<comm>.*?)\s+(?P<pid>-?\d+)/(?P<tid>-?\d+)\s+\[\s*(?P<cpu>\d+)\s*\]\s+"
                     r"(?P<sec>\d+)\.(?P<frac>\d+):\s*(?:[\w.-]+:)?"
                     r"(?P<ev>irq_handler_entry|irq_handler_exit|softirq_entry|softirq_exit):\s*(?P<trace>.*?)\s*$")
CPU_RE = re.compile(r"\[\s*(\d+)\s*\]")
PIDTID_RE = re.compile(r"(?<!\S)(-?\d+)/(-?\d+)(?!\S)")
LONE_INT_RE = re.compile(r"(?<!\S)(-?\d+)(?!\S)")
HARD_ENTRY_RE = re.compile(r"irq=(-?\d+)(?:\s+name=(.*))?$")
HARD_EXIT_RE = re.compile(r"irq=(-?\d+)(?:\s+ret=(\S+))?")
SOFT_RE = re.compile(r"vec=(\d+)(?:\s+\[action=([^\]]*)\])?")

log = logging.getLogger("irq_time_window")


@dataclass(slots=True)
class Event:
    ts_ns: int
    cpu: int
    pid: int | None
    tid: int | None
    comm: str
    kind: str        # HARD | SOFT
    is_entry: bool
    key: int         # irq number or softirq vec
    label: str       # irq name (entry only) / softirq action / "" when absent
    ret: str = ""    # hardirq exit: handled | unhandled
    seq: int = 0     # input order, keeps sorting stable

    @property
    def idle(self) -> bool:
        if self.pid is not None:
            return self.pid == 0
        if self.tid is not None:
            return self.tid == 0
        return self.comm.startswith("swapper")


@dataclass(slots=True)
class Frame:
    kind: str
    key: int
    start_ns: int
    idle: bool
    label: str
    child_ns: int = 0


@dataclass(slots=True)
class Done:
    kind: str
    key: int
    label: str
    cpu: int
    idle: bool
    self_ns: int
    incl_ns: int
    nested: bool


@dataclass(slots=True)
class ParseStats:
    lines: int = 0
    events: int = 0
    skipped_lines: int = 0          # blank, comment, other events
    bad_lines: list[str] = field(default_factory=list)
    bad_count: int = 0


def _ts_ns(sec: str, frac: str) -> int:
    return int(sec) * 1_000_000_000 + int(frac[:9].ljust(9, "0"))


def _split_prefix(pre: str) -> tuple[int | None, int | None, int | None, str]:
    """Return (cpu, pid, tid, comm) from the text before the timestamp."""
    m = CPU_RE.search(pre)
    if m is None:
        return None, None, None, pre.strip()
    cpu = int(m.group(1))
    rest = pre[:m.start()] + " " + pre[m.end():]
    pid = tid = None
    comm_end = len(rest)
    pts = list(PIDTID_RE.finditer(rest))
    if pts:
        last = pts[-1]
        pid, tid = int(last.group(1)), int(last.group(2))
        comm_end = last.start()
    else:
        ints = list(LONE_INT_RE.finditer(rest))
        if ints and ints[-1].start() > 0:   # a bare number with a comm before it: -F pid or -F tid alone
            pid = tid = int(ints[-1].group(1))
            comm_end = ints[-1].start()
    return cpu, pid, tid, rest[:comm_end].strip()


def parse_line(line: str, seq: int = 0) -> Event | None | str:
    """Event for an irq tracepoint line, None for an unrelated line, or an error string for a malformed one."""
    s = line.rstrip("\n")
    if not s.strip() or s.lstrip().startswith("#"):
        return None
    m = FAST_RE.match(s)
    if m is not None:
        cpu, pid, tid, comm = int(m.group("cpu")), int(m.group("pid")), int(m.group("tid")), m.group("comm")
    else:
        m = LINE_RE.match(s)
        if m is None:
            return None
        cpu, pid, tid, comm = _split_prefix(m.group("pre"))
        if cpu is None:
            return "no [cpu] field"
    ev, trace = m.group("ev"), m.group("trace")
    ts = _ts_ns(m.group("sec"), m.group("frac"))
    if ev == "irq_handler_entry":
        t = HARD_ENTRY_RE.search(trace)
        if t is None:
            return "bad irq_handler_entry trace"
        return Event(ts, cpu, pid, tid, comm, HARD, True, int(t.group(1)), (t.group(2) or "").strip(), seq=seq)
    if ev == "irq_handler_exit":
        t = HARD_EXIT_RE.search(trace)
        if t is None:
            return "bad irq_handler_exit trace"
        return Event(ts, cpu, pid, tid, comm, HARD, False, int(t.group(1)), "", t.group(2) or "", seq=seq)
    t = SOFT_RE.search(trace)
    if t is None:
        return f"bad {ev} trace"
    vec = int(t.group(1))
    label = (t.group(2) or "").strip() or (SOFTIRQ_NAMES[vec] if vec < len(SOFTIRQ_NAMES) else "")
    return Event(ts, cpu, pid, tid, comm, SOFT, ev == "softirq_entry", vec, label, seq=seq)


def parse_text(lines, stats: ParseStats) -> list[Event]:
    events: list[Event] = []
    for i, line in enumerate(lines):
        stats.lines += 1
        r = parse_line(line, i)
        if r is None:
            stats.skipped_lines += 1
        elif isinstance(r, str):
            stats.bad_count += 1
            if len(stats.bad_lines) < 5:
                stats.bad_lines.append(f"line {i + 1}: {r}: {line.strip()[:160]}")
        else:
            events.append(r)
    stats.events = len(events)
    return events


@dataclass(slots=True)
class Pairing:
    done: list[Done] = field(default_factory=list)
    unmatched: dict[str, int] = field(default_factory=lambda: {"hardirq_entry": 0, "hardirq_exit": 0,
                                                               "softirq_entry": 0, "softirq_exit": 0})
    negative: int = 0
    max_depth: int = 0
    unhandled: dict[int, int] = field(default_factory=dict)
    irq_names: dict[int, str] = field(default_factory=dict)


def pair_events(events: list[Event]) -> Pairing:
    """Pair entries and exits per CPU on a stack; record self/inclusive durations."""
    out = Pairing()
    stacks: dict[int, list[Frame]] = {}
    for e in sorted(events, key=lambda x: (x.ts_ns, x.seq)):
        st = stacks.setdefault(e.cpu, [])
        if e.is_entry:
            if e.kind == HARD and e.label:
                out.irq_names.setdefault(e.key, e.label)
            # The same irq/vec still open on this CPU means its exit was lost (they do not re-enter).
            idx = _find(st, e.kind, e.key)
            if idx is not None:
                _drop_above(st, idx - 1, out)
            st.append(Frame(e.kind, e.key, e.ts_ns, e.idle, e.label))
            out.max_depth = max(out.max_depth, len(st))
            continue
        idx = _find(st, e.kind, e.key)
        if idx is None:
            out.unmatched[f"{e.kind}_exit"] += 1
            continue
        _drop_above(st, idx, out)            # frames above it lost their exits
        fr = st.pop()
        incl = e.ts_ns - fr.start_ns
        if incl < 0:
            out.negative += 1
            continue
        self_ns = max(incl - fr.child_ns, 0)
        if st:
            st[-1].child_ns += incl
        if fr.kind == HARD and e.ret == "unhandled":
            out.unhandled[fr.key] = out.unhandled.get(fr.key, 0) + 1
        out.done.append(Done(fr.kind, fr.key, fr.label, e.cpu, fr.idle, self_ns, incl, bool(st)))
    for st in stacks.values():
        for fr in st:
            out.unmatched[f"{fr.kind}_entry"] += 1
    return out


def _find(st: list[Frame], kind: str, key: int) -> int | None:
    for i in range(len(st) - 1, -1, -1):
        if st[i].kind == kind and st[i].key == key:
            return i
    return None


def _drop_above(st: list[Frame], idx: int, out: Pairing) -> None:
    """Discard frames above index idx as unmatched entries (their exit is missing)."""
    while len(st) > idx + 1:
        fr = st.pop()
        out.unmatched[f"{fr.kind}_entry"] += 1


def _pct(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolated percentile (numpy default) of an ascending list."""
    if not sorted_vals:
        return math.nan
    pos = (len(sorted_vals) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def _ms(ns: int) -> float:
    return round(ns / 1e6, 6)


def _us(v: float) -> float | None:
    return None if math.isnan(v) else round(v, 3)


def _group_stats(items: list[Done]) -> dict:
    durs = sorted(d.self_ns / 1e3 for d in items)
    return {
        "count": len(items),
        "total_ms": _ms(sum(d.self_ns for d in items)),
        "idle_ms": _ms(sum(d.self_ns for d in items if d.idle)),
        "median_us": _us(_pct(durs, 50)),
        "p99_us": _us(_pct(durs, 99)),
        "max_us": _us(durs[-1]) if durs else None,
    }


def _kind_summary(items: list[Done]) -> dict:
    total = sum(d.self_ns for d in items)
    idle = sum(d.self_ns for d in items if d.idle)
    return {
        "count": len(items),
        "total_ms": _ms(total),
        "idle_count": sum(1 for d in items if d.idle),
        "idle_ms": _ms(idle),
        "busy_ms": _ms(total - idle),
        "incl_ms": _ms(sum(d.incl_ns for d in items)),
        "nested_count": sum(1 for d in items if d.nested),
    }


def summarize(events: list[Event], pairing: Pairing, frames: int | None) -> dict:
    res: dict = {}
    if events:
        res["window_s"] = round((max(e.ts_ns for e in events) - min(e.ts_ns for e in events)) / 1e9, 9)
    else:
        res["window_s"] = None
    hard = [d for d in pairing.done if d.kind == HARD]
    soft = [d for d in pairing.done if d.kind == SOFT]
    res["hardirq"] = _kind_summary(hard)
    res["softirq"] = _kind_summary(soft)

    by_irq: dict[str, dict] = {}
    for key in sorted({d.key for d in hard}):
        items = [d for d in hard if d.key == key]
        name = pairing.irq_names.get(key, "")
        st = {"irq": key, "name": name, **_group_stats(items)}
        if pairing.unhandled.get(key):
            st["unhandled"] = pairing.unhandled[key]
        by_irq[f"{key}:{name}" if name else str(key)] = st
    res["by_irq"] = dict(sorted(by_irq.items(), key=lambda kv: -kv[1]["total_ms"]))

    by_soft: dict[str, dict] = {}
    for key in sorted({d.key for d in soft}):
        items = [d for d in soft if d.key == key]
        label = next((d.label for d in items if d.label), "")
        by_soft[f"{key}:{label}" if label else str(key)] = {"vec": key, "action": label, **_group_stats(items)}
    res["by_softirq"] = dict(sorted(by_soft.items(), key=lambda kv: -kv[1]["total_ms"]))

    by_cpu: dict[str, dict] = {}
    for cpu in sorted({d.cpu for d in pairing.done}):
        h = [d for d in hard if d.cpu == cpu]
        s = [d for d in soft if d.cpu == cpu]
        by_cpu[str(cpu)] = {
            "hardirq_ms": _ms(sum(d.self_ns for d in h)),
            "hardirq_idle_ms": _ms(sum(d.self_ns for d in h if d.idle)),
            "softirq_ms": _ms(sum(d.self_ns for d in s)),
            "softirq_idle_ms": _ms(sum(d.self_ns for d in s if d.idle)),
        }
    res["by_cpu"] = by_cpu

    um = dict(pairing.unmatched)
    um["total"] = sum(pairing.unmatched.values())
    res["unmatched"] = um
    res["negative_durations"] = pairing.negative
    res["max_nesting_depth"] = pairing.max_depth

    uncounted_ns = sum(d.self_ns for d in pairing.done if d.idle)
    res["uncounted_ms"] = _ms(uncounted_ns)
    res["frames"] = frames
    if frames and pairing.done:
        for kind, s in ((HARD, res[HARD]), (SOFT, res[SOFT])):
            for k in ("total_ms", "idle_ms", "busy_ms"):
                res[f"{kind}_{k.removesuffix('_ms')}_ms_per_frame"] = round(s[k] / frames, 6)
        res["uncounted_ms_per_frame"] = round(uncounted_ns / 1e6 / frames, 6)
    else:
        res["uncounted_ms_per_frame"] = None
    return res


def csv_line(res: dict | None) -> str:
    """One CSV row in CSV_FIELDS order; all nan when there is no paired event."""
    if not res or (res["hardirq"]["count"] + res["softirq"]["count"]) == 0:
        return ",".join(["nan"] * len(CSV_FIELDS))

    def f(v, fmt: str) -> str:
        return "nan" if v is None or (isinstance(v, float) and math.isnan(v)) else format(v, fmt)

    h, s = res["hardirq"], res["softirq"]
    vals = (f(res["window_s"], ".3f"), str(h["count"]), f(h["total_ms"], ".3f"), f(h["idle_ms"], ".3f"),
            str(s["count"]), f(s["total_ms"], ".3f"), f(s["idle_ms"], ".3f"), str(res["unmatched"]["total"]),
            f(res["uncounted_ms_per_frame"], ".4f"))
    return ",".join(vals)


def analyze(script: Path, frames: int | None) -> dict:
    """Parse and summarize; never raises -- failures land in result["error"]."""
    stats = ParseStats()
    res: dict | None = None
    error: str | None = None
    events: list[Event] = []
    try:
        with script.open("r", encoding="utf-8", errors="replace") as fh:
            events = parse_text(fh, stats)
        res = summarize(events, pair_events(events), frames)
    except OSError as exc:
        error = f"cannot read {script}: {exc}"
    except Exception as exc:  # keep the bench running: record the failure instead of a traceback
        error = f"{type(exc).__name__}: {exc}"
        log.exception("analysis failed")
    if res is None:
        res = summarize([], Pairing(), frames)
    if error is None and stats.events == 0:
        error = "no irq tracepoint events in input"
    out = {"tool": "irq_time_window", "script": str(script), **res,
           "parse": {"lines": stats.lines, "irq_events": stats.events, "skipped_lines": stats.skipped_lines,
                     "bad_lines": stats.bad_count, "bad_examples": stats.bad_lines},
           "error": error}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Hardirq/softirq time (total, idle share) from irq tracepoints in "
                                            "'perf script' text; JSON to --out, one CSV line to stdout.")
    p.add_argument("--script", type=Path, help="perf script text (-F cpu,pid,tid,comm,time,event,trace)")
    p.add_argument("--frames", type=int, default=None, help="frames in the window (adds *_ms_per_frame)")
    p.add_argument("--out", type=Path, help="output JSON path")
    p.add_argument("--csv-header", action="store_true", help="print the CSV header line only and exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="irq_time_window: %(levelname)s: %(message)s")

    if args.csv_header:
        print(",".join(CSV_FIELDS))
        return EXIT_OK
    if args.script is None or args.out is None:
        log.error("--script and --out are required (or --csv-header)")
        print(csv_line(None))
        return EXIT_USAGE

    frames = args.frames
    res = analyze(args.script, frames if frames and frames > 0 else None)
    if frames is not None and frames <= 0:
        res["warning"] = f"--frames {frames} ignored (must be > 0)"
    if res["error"]:
        log.warning("%s", res["error"])

    rc = EXIT_OK if res["error"] is None and res["hardirq"]["count"] + res["softirq"]["count"] > 0 else EXIT_PARTIAL
    try:
        with args.out.open("w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.write("\n")
    except (OSError, ValueError) as exc:
        log.error("cannot write %s: %s", args.out, exc)
        rc = EXIT_USAGE
    print(csv_line(res))
    return rc


if __name__ == "__main__":
    sys.exit(main())
