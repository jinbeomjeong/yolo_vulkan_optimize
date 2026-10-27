#!/usr/bin/env python3
"""Summarize a memasym_trace.sh capture (<out_dir>/probe.txt + trace.txt) into plain text.

1. PTE protection per CPU mapping: kprobe events vmf_insert_pfn_prot / remap_pfn_range whose user address falls in a
   "[map] <name> ptr <p> len <n>" range of the probe log; prot -> arm64 AttrIndx (bits [4:2]) -> MT_* name.
2. Coherence windows: for every "[mark] mono=<t> coherence <alloc> <dir> sync=<s> trial=<k> <phase>" the events
   between this mark and the next one (trace_clock mono = CLOCK_MONOTONIC), counted per event and task, plus the
   kernel stacks of the cache-maintenance events (arch_sync_dma_*; the dcache op is inferred from the direction).

Usage: python3 06_tools/probes/memasym_trace_summary.py <out_dir>   (reads trace.txt or trace.txt.gz)
"""
import collections
import gzip
import re
import sys
from pathlib import Path

MT = {0: "MT_NORMAL (write-back cacheable)", 1: "MT_NORMAL_TAGGED", 2: "MT_NORMAL_NC (non-cacheable)",
      3: "MT_DEVICE_nGnRnE", 4: "MT_DEVICE_nGnRE"}
CACHE_EVENTS = ("arch_sync_dma_for_device", "arch_sync_dma_for_cpu")
# enum dma_data_direction -> CPU cache operation (arch/arm64/mm/dma-mapping.c, rpi-6.18.y)
DEV_OP = {0: "clean_inval", 1: "clean", 2: "inval"}
DIR_NAME = {0: "BIDIRECTIONAL", 1: "TO_DEVICE", 2: "FROM_DEVICE", 3: "NONE"}
PROBE_COMM = "yvo_postproc_pr"   # TASK_COMM_LEN 16 truncates yvo_postproc_probe

EV_RE = re.compile(r"^\s*(?P<task>.+?)-(?P<pid>\d+)\s+(?:\(\s*(?:\d+|-+)\)\s+)?\[(?P<cpu>\d+)\]\s+\S+\s+"
                   r"(?P<ts>\d+\.\d+):\s+(?P<ev>\w+):\s+\((?P<loc>[^)]*)\)\s*(?P<args>.*)$")
STACK_HDR_RE = re.compile(r"^\s*(?P<task>.+?)-(?P<pid>\d+).*?(?P<ts>\d+\.\d+):\s+<stack trace>")
ARG_RE = re.compile(r"(\w+)=(0x[0-9a-fA-F]+|-?\d+)")


def parse_probe(path):
    """maps: (name, base, len, t_begin, t_end) where [t_begin, t_end) is the enclosing [part] window."""
    maps, marks, parts = [], [], []
    pending = []
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"\[part\] mono=(\d+\.\d+) (begin|end)", line)
        if m:
            t = float(m.group(1))
            for name, base, length, t0 in pending:
                maps.append((name, base, length, t0, t))
            pending = []
            parts.append(t)
            continue
        m = re.match(r"\[map\] (\S+) ptr (0x[0-9a-f]+) len (\d+)", line)
        if m:
            pending.append((m.group(1), int(m.group(2), 16), int(m.group(3)), parts[-1] if parts else 0.0))
            continue
        m = re.match(r"\[mark\] mono=(\d+\.\d+) coherence (\S+) (\S+) sync=(\S+) trial=(\d+) (\S+)", line)
        if m:
            marks.append((float(m.group(1)), m.group(2), m.group(3), m.group(4), int(m.group(5)), m.group(6)))
    return maps, marks


def parse_trace(path):
    events = []   # dict(ts, ev, task, pid, args, stack)
    cur = None
    text = gzip.open(path, "rt", errors="replace").read() if path.suffix == ".gz" else path.read_text(errors="replace")
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        m = EV_RE.match(line)
        if m and m.group("ev") != "":
            args = {k: int(v, 16) if v.startswith("0x") else int(v) for k, v in ARG_RE.findall(m.group("args"))}
            cur = {"ts": float(m.group("ts")), "ev": m.group("ev"), "task": m.group("task").strip(),
                   "pid": int(m.group("pid")), "args": args, "stack": []}
            events.append(cur)
            continue
        if STACK_HDR_RE.match(line):
            continue   # the stack lines that follow belong to the previous event
        s = line.strip()
        if s.startswith("=>") and cur is not None:
            cur["stack"].append(s[2:].strip())
    return events


KPROBE_FRAMES = ("kprobe_", "call_el1_break_hook", "do_el1_brk64", "el1_brk64", "el1h_64_sync")


def caller_frames(stack, n=8):
    """Drop the kprobe/breakpoint machinery at the top of a tracefs stack and keep the next n function names."""
    names = [f.split("+")[0] for f in stack]
    i = 0
    while i < len(names) and names[i].startswith(KPROBE_FRAMES):
        i += 1
    return names[i:i + n]


def main():
    out = Path(sys.argv[1])
    maps, marks = parse_probe(out / "probe.txt")
    events = parse_trace(out / "trace.txt" if (out / "trace.txt").exists() else out / "trace.txt.gz")
    print(f"# memasym trace summary: {out}")
    print(f"events {len(events)}, probe maps {len(maps)}, marks {len(marks)}")
    by_ev = collections.Counter(e["ev"] for e in events)
    print("event totals: " + ", ".join(f"{k} {v}" for k, v in sorted(by_ev.items())))

    print("\n## 1. PTE protection of the probe's CPU mappings")
    print("map      range                                 event                 n_pages  prot values -> AttrIndx")
    for name, base, length, tb, te in maps:
        hits = collections.Counter()
        evname = collections.Counter()
        npages = 0
        for e in events:
            if e["ev"] not in ("vmf_insert_pfn_prot", "remap_pfn_range") or not e["task"].startswith(PROBE_COMM):
                continue
            a = e["args"].get("addr")
            if a is None or not (base <= a < base + length) or not (tb <= e["ts"] < te):
                continue
            hits[e["args"]["prot"]] += 1
            evname[e["ev"]] += 1
            npages += 1
        desc = "; ".join(f"0x{p:x} -> {(p >> 2) & 7} {MT.get((p >> 2) & 7, '?')} (x{c})" for p, c in hits.items()) \
            or "no event in range (malloc / already mapped)"
        print(f"{name:7s}  0x{base:x}-0x{base + length:x}  {','.join(evname) or '-':20s}  {npages:7d}  {desc}")

    print("\n## 2. Events inside each coherence window ([mark] .. next [mark])")
    marks.sort()
    windows = collections.OrderedDict()
    for i, (t0, alloc, d, sync, trial, phase) in enumerate(marks):
        t1 = marks[i + 1][0] if i + 1 < len(marks) else float("inf")
        key = (alloc, d, sync, phase)
        w = windows.setdefault(key, {"n": 0, "dur": 0.0, "ev": collections.Counter(), "bytes": collections.Counter(),
                                     "stacks": collections.Counter()})
        w["n"] += 1
        w["dur"] += (t1 - t0) if t1 != float("inf") else 0.0
        for e in events:
            if t0 <= e["ts"] < t1:
                tag = e["ev"]
                if e["ev"] in CACHE_EVENTS:
                    d = e["args"].get("dir", -1)
                    op = DEV_OP.get(d, "none") if e["ev"] == "arch_sync_dma_for_device" else \
                        ("none" if d == 1 else "inval")
                    tag += f"({DIR_NAME.get(d, d)} -> dcache_{op})"
                tag += "" if e["task"].startswith(PROBE_COMM) else f"[{e['task']}]"
                w["ev"][tag] += 1
                if e["ev"] in CACHE_EVENTS:
                    w["bytes"][tag] += e["args"].get("size", 0)
                if e["ev"] in CACHE_EVENTS and e["stack"]:
                    w["stacks"][(e["ev"], " <- ".join(caller_frames(e["stack"])))] += 1
    rest = {"gpu-done": "rest of the cell: CPU check + next cell setup", "cpu-done": "rest of the cell: check + next setup"}
    for (alloc, d, sync, phase), w in windows.items():
        label = f"  ({rest[phase]})" if phase in rest else ""
        print(f"\n[{alloc} {d} sync={sync} {phase}]  windows {w['n']}, total {w['dur'] * 1e3:.2f} ms{label}")
        if not w["ev"]:
            print("  (no traced event)")
        for tag, c in sorted(w["ev"].items()):
            extra = f"  bytes {w['bytes'][tag]}" if tag in w["bytes"] else ""
            print(f"  {tag:60s} {c:6d}{extra}")
        for (ev, st), c in w["stacks"].most_common(4):
            print(f"  stack x{c} {ev}: {st}")


if __name__ == "__main__":
    main()
