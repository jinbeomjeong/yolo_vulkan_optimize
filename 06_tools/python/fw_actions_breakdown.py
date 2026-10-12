#!/usr/bin/env python3
"""Per-context time breakdown of HailoRT measure-fw-actions JSON, and T(B) = a + b*B fits across batch sizes.
Stdlib only.

Input: the JSON written by
  hailortcli run2 -t 3 -m raw_async measure-fw-actions --output-path X.json set-net <hef> --batch-size B
Schema as found in the 2026-10-08 probe files (HailoRT 4.24.0, "version": "2.0"):
  {clock_cycle_MHz: 200, hef: {path, file_hash},
   runs: [{batch_size, fps, mean_activation_time_ms, mean_deactivation_time_ms, network_group_id,
           contexts: [{context_name, batch_counter, idle_time, action_list_base_address, action_list_size,
                       actions: [{address, timestamp, type, data, [sub_action_index]}]}]}]}
  context_name: activation, preliminary, dynamic_0..N-1, batch_switching.
  timestamp: firmware cycle counter (clock_cycle_MHz) at which the action was recorded. Its origin is not
  documented; only differences are used here. batch_switching carries all-zero timestamps (not timed).
  Each context lists its action list once: the timestamps are from one recorded execution (inferred: the last
  one), while batch_counter counts how many times the context ran (inferred). So one file gives one batch
  iteration. If a context name repeats inside a run, the repeats are reported as separate iterations
  ("name#2", ...); several runs are reported separately.

Definitions (all in ms, cycles / (clock_cycle_MHz * 1000)):
  span          last timestamp - first timestamp of a context.
  type time     the interval from the previous action's timestamp to this action's timestamp is charged to
                this action's type ("time spent waiting until this action was recorded"). The type times of
                a context therefore add up exactly to its span. E.g. the wait before an lcu_interrupt is
                (inferred) compute of that LCU's layers, possibly overlapped with config/data DMA.
  switch gap    first timestamp of dynamic_{i+1} - last timestamp of dynamic_i (context switch overhead).
  batch span    last timestamp of the last dynamic context - first timestamp of dynamic_0 (= sum of dynamic
                spans + switch gaps), when the dynamic contexts are recorded back to back (checked).
  fps period    1000 * batch_size / fps, the run's average batch period reported by hailortcli.
Interpretations of action names are not officially documented and are marked "inferred" (see TYPE_NOTES).

Fit (when files with at least two distinct batch sizes are given; repeats of a B are all used as points):
  least squares span(B) = a + b*B per dynamic context, for the dynamic span sum, the batch span and the fps
  period, plus per (context, action type) for types above --fit-min-ms. Compared with the pipeline model
  T(B) = 24.8 + 16.2*B ms (07_results/op_point; override with --ref A,B).

Usage:
  python3 06_tools/python/fw_actions_breakdown.py X_b1.json X_b8.json [more.json ...]
      [--top-types 6] [--gaps 3] [--fit-min-ms 0.5] [--ref 24.8,16.2] [--json OUT.json]
Text goes to stdout; --json writes the parsed breakdown and fits. Output paths equal to an input are refused.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

WRAP = 1 << 32                 # assumed counter width when a timestamp goes backwards (inferred, never seen so far)
REF_MODEL = (24.8, 16.2)       # pipeline T(B) = a + b*B ms (07_results/op_point)
DYN_RE = re.compile(r"^dynamic_(\d+)$")

# Short readings of action types. None of these is officially documented: all are inferred from names and timing.
TYPE_NOTES = {
    "lcu_interrupt": "an LCU (layer control unit) reported done -> wait before it ~ compute (inferred)",
    "fetch_cfg_channel_descriptors": "start config (weights/CCW) DMA on a cfg channel (inferred)",
    "module_config_done_interrupt": "a config module finished loading (inferred)",
    "trigger_sequencer": "start a config sequencer (inferred)",
    "sequencer_done_interrupt": "config sequencer finished (inferred)",
    "enable_lcu_default": "program/enable an LCU for this context (inferred)",
    "disable_lcu": "disable an LCU after its interrupt (inferred)",
    "output_channel_transfer_done_interrupt": "an output DMA transfer finished (inferred)",
    "deactivate_vdma_channel": "close a vDMA channel at context end (inferred)",
    "activate_cfg_channel": "open a config channel (inferred)",
    "fetch_data_from_vdma_channel": "start input data DMA (inferred)",
}


class FwActionsError(ValueError):
    """The JSON does not have the measure-fw-actions structure."""


@dataclass
class Context:
    name: str
    batch_counter: int | None
    idle_time: int | None
    actions: list[tuple[int, str]]          # (timestamp cycles, type), wrap-corrected, in file order
    wraps: int = 0

    @property
    def timed(self) -> bool:
        return any(t != 0 for t, _ in self.actions)


@dataclass
class Run:
    file: str
    index: int
    batch_size: int
    fps: float | None
    clock_mhz: float
    contexts: list[Context] = field(default_factory=list)

    def ms(self, cycles: float) -> float:
        return cycles / (self.clock_mhz * 1000.0)


def load(path: str | Path) -> list[Run]:
    """Parse one measure-fw-actions JSON into runs."""
    with open(path, encoding="utf-8") as fh:
        try:
            d = json.load(fh)
        except json.JSONDecodeError as e:
            raise FwActionsError(f"{path}: not JSON ({e})") from e
    if not isinstance(d, dict) or "runs" not in d or "clock_cycle_MHz" not in d:
        raise FwActionsError(f"{path}: missing 'runs' or 'clock_cycle_MHz' (not a measure-fw-actions JSON)")
    clk = float(d["clock_cycle_MHz"])
    if clk <= 0:
        raise FwActionsError(f"{path}: clock_cycle_MHz must be > 0")
    runs = []
    for ri, r in enumerate(d["runs"]):
        try:
            run = Run(str(path), ri, int(r["batch_size"]), r.get("fps"), clk)
            seen: Counter[str] = Counter()
            for c in r["contexts"]:
                name = c["context_name"]
                seen[name] += 1
                if seen[name] > 1:
                    name = f"{name}#{seen[name]}"
                acts, wraps, off, prev = [], 0, 0, None
                for a in c["actions"]:
                    t = int(a["timestamp"]) + off
                    if prev is not None and t < prev:
                        off += WRAP
                        t += WRAP
                        wraps += 1
                    acts.append((t, str(a["type"])))
                    prev = t
                run.contexts.append(Context(name, c.get("batch_counter"), c.get("idle_time"), acts, wraps))
        except (KeyError, TypeError, ValueError) as e:
            raise FwActionsError(f"{path}: run {ri}: unexpected structure ({e!r})") from e
        runs.append(run)
    return runs


# ---------------------------------------------------------------- per-run analysis

def type_times(ctx: Context) -> Counter[str]:
    """Cycles charged to each action type (interval prev -> this charged to this). Sums to the span."""
    out: Counter[str] = Counter()
    for (t0, _), (t1, typ) in zip(ctx.actions, ctx.actions[1:]):
        out[typ] += t1 - t0
    return out


def largest_gaps(ctx: Context, k: int) -> list[tuple[int, int, str, str]]:
    """The k largest intervals: (cycles, index of the later action, previous type, this type)."""
    gaps = [(t1 - t0, i + 1, a0, a1) for i, ((t0, a0), (t1, a1)) in enumerate(zip(ctx.actions, ctx.actions[1:]))]
    return sorted(gaps, key=lambda g: (-g[0], g[1]))[:k]


def dynamic_contexts(run: Run) -> list[Context]:
    dyn = [c for c in run.contexts if DYN_RE.match(c.name)]
    return sorted(dyn, key=lambda c: int(DYN_RE.match(c.name).group(1)))


def analyze_run(run: Run, gaps_k: int = 3) -> dict:
    ctxs = []
    for c in run.contexts:
        ts = [t for t, _ in c.actions]
        tt = type_times(c)
        ctxs.append({
            "name": c.name, "timed": c.timed, "actions": len(c.actions), "batch_counter": c.batch_counter,
            "idle_time_ms": None if c.idle_time is None else run.ms(c.idle_time), "wraps": c.wraps,
            "first_cycles": min(ts) if ts else None, "last_cycles": max(ts) if ts else None,
            "span_ms": run.ms(max(ts) - min(ts)) if ts and c.timed else None,
            "type_ms": {k: run.ms(v) for k, v in sorted(tt.items(), key=lambda kv: -kv[1])} if c.timed else {},
            "type_counts": dict(Counter(typ for _, typ in c.actions)),
            "largest_gaps": [{"ms": run.ms(g), "index": i, "after": a0, "before": a1}
                             for g, i, a0, a1 in largest_gaps(c, gaps_k)] if c.timed else [],
        })
    dyn = dynamic_contexts(run)
    switch = []
    contiguous = bool(dyn)
    for a, b in zip(dyn, dyn[1:]):
        g = b.actions[0][0] - a.actions[-1][0] if a.actions and b.actions else None
        switch.append(None if g is None else run.ms(g))
        if g is None or g < 0:
            contiguous = False
    by = {c["name"]: c for c in ctxs}
    dyn_sum = sum(by[c.name]["span_ms"] or 0.0 for c in dyn)
    batch_span = run.ms(dyn[-1].actions[-1][0] - dyn[0].actions[0][0]) if contiguous else None
    return {
        "file": run.file, "run": run.index, "batch_size": run.batch_size, "fps": run.fps,
        "clock_MHz": run.clock_mhz, "contexts": ctxs, "dynamic": [c.name for c in dyn],
        "switch_gaps_ms": switch, "contiguous": contiguous,
        "dynamic_span_sum_ms": dyn_sum, "batch_span_ms": batch_span,
        "fps_period_ms": 1000.0 * run.batch_size / run.fps if run.fps else None,
    }


# ---------------------------------------------------------------- fits

def fit_affine(points: list[tuple[float, float]]) -> dict | None:
    """Least squares y = a + b*x. None unless there are >= 2 distinct x. rms = residual RMS (0 for 2 points)."""
    xs = [x for x, _ in points]
    if len(set(xs)) < 2:
        return None
    n = len(points)
    mx = sum(xs) / n
    my = sum(y for _, y in points) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    rms = (sum((y - a - b * x) ** 2 for x, y in points) / n) ** 0.5
    return {"a": a, "b": b, "n": n, "rms": rms}


def fit_all(results: list[dict], min_ms: float) -> dict:
    """Fits over all analyzed runs: per dynamic context, per (context, type), and totals."""
    ctx_pts: dict[str, list] = defaultdict(list)
    type_pts: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    tot: dict[str, list] = defaultdict(list)
    for ri, r in enumerate(results):
        B = r["batch_size"]
        for c in r["contexts"]:
            if c["name"] in r["dynamic"] and c["span_ms"] is not None:
                ctx_pts[c["name"]].append((B, c["span_ms"]))
                for typ, v in c["type_ms"].items():
                    type_pts[(c["name"], typ)][ri] = v
        tot["dynamic_span_sum"].append((B, r["dynamic_span_sum_ms"]))
        if r["batch_span_ms"] is not None:
            tot["batch_span"].append((B, r["batch_span_ms"]))
        if r["fps_period_ms"] is not None:
            tot["fps_period"].append((B, r["fps_period_ms"]))
    out: dict = {"contexts": {}, "types": {}, "totals": {}}
    for name, pts in sorted(ctx_pts.items(), key=lambda kv: int(DYN_RE.match(kv[0]).group(1))):
        out["contexts"][name] = fit_affine(pts)
    for (name, typ), per_run in type_pts.items():
        # a type missing from a run's context had 0 time there
        pts = [(r["batch_size"], per_run.get(ri, 0.0)) for ri, r in enumerate(results) if name in r["dynamic"]]
        if max(y for _, y in pts) >= min_ms:
            f = fit_affine(pts)
            if f:
                out["types"].setdefault(name, {})[typ] = f
    for k, pts in tot.items():
        out["totals"][k] = fit_affine(pts)
    return out


# ---------------------------------------------------------------- output

def _f(v: float | None, nd: int = 2) -> str:
    return "n/a" if v is None else f"{v:.{nd}f}"


def format_run(r: dict, top_types: int) -> str:
    L = [f"=== {r['file']} run {r['run']}: batch {r['batch_size']}, fps {_f(r['fps'])}, "
         f"fps period {_f(r['fps_period_ms'])} ms (clock {r['clock_MHz']:g} MHz)"]
    L.append(f"  {'context':<16}{'actions':>8}{'span ms':>10}{'batch_ctr':>10}{'idle_time ms':>14}")
    for c in r["contexts"]:
        span = _f(c["span_ms"]) if c["timed"] else "not timed"
        L.append(f"  {c['name']:<16}{c['actions']:>8}{span:>10}{_f(c['batch_counter'], 0) if c['batch_counter'] is not None else 'n/a':>10}"
                 f"{_f(c['idle_time_ms']):>14}" + (f"  ({c['wraps']} counter wraps assumed)" if c["wraps"] else ""))
    gaps = ", ".join(_f(g, 3) for g in r["switch_gaps_ms"])
    L.append(f"  dynamic span sum {_f(r['dynamic_span_sum_ms'])} ms; switch gaps [{gaps}] ms; "
             + (f"batch span {_f(r['batch_span_ms'])} ms" if r["contiguous"]
                else "dynamic contexts not back to back (no batch span)")
             + f"; fps period {_f(r['fps_period_ms'])} ms")
    L.append("  time by action type (interval before an action charged to it; sums to the span):")
    for c in r["contexts"]:
        if not c["timed"]:
            continue
        items = list(c["type_ms"].items())
        shown = items[:top_types] if top_types > 0 else items
        rest = sum(v for _, v in items[len(shown):])
        parts = [f"{k} {_f(v)}" for k, v in shown] + ([f"other {_f(rest)}"] if items[len(shown):] else [])
        L.append(f"    {c['name']:<14} " + ", ".join(parts))
        for g in c["largest_gaps"]:
            if g["ms"] < 0.01:      # sub-10 us intervals are action bookkeeping, not worth a line
                continue
            L.append(f"      gap {_f(g['ms']):>7} ms before #{g['index']} {g['before']} (after {g['after']})")
    return "\n".join(L) + "\n"


def format_fits(fits: dict, results: list[dict], ref: tuple[float, float]) -> str:
    Bs = sorted({r["batch_size"] for r in results})
    L = [f"=== fit span(B) = a + b*B over B = {Bs} ({len(results)} runs); reference T(B) = {ref[0]:g} + {ref[1]:g}*B"]
    L.append(f"  {'series':<22}{'a ms':>8}{'b ms/B':>8}{'n':>4}{'rms':>7}  " + "".join(f"{'B=' + str(b):>9}" for b in Bs))

    def row(name: str, f: dict | None) -> str:
        if f is None:
            return f"  {name:<22}  (needs >= 2 distinct B)"
        return (f"  {name:<22}{f['a']:>8.2f}{f['b']:>8.2f}{f['n']:>4}{f['rms']:>7.2f}  "
                + "".join(f"{f['a'] + f['b'] * b:>9.2f}" for b in Bs))
    for name, f in fits["contexts"].items():
        L.append(row(name, f))
    a_sum = sum(f["a"] for f in fits["contexts"].values() if f)
    b_sum = sum(f["b"] for f in fits["contexts"].values() if f)
    if fits["contexts"] and all(fits["contexts"].values()):
        L.append(f"  {'sum of context fits':<22}{a_sum:>8.2f}{b_sum:>8.2f}")
    for k in ("dynamic_span_sum", "batch_span", "fps_period"):
        if k in fits["totals"]:
            L.append(row(k, fits["totals"][k]))
    L.append(f"  {'reference (pipeline)':<22}{ref[0]:>8.2f}{ref[1]:>8.2f}{'':>11}  "
             + "".join(f"{ref[0] + ref[1] * b:>9.2f}" for b in Bs))
    if fits["types"]:
        L.append("  per (context, action type) fits (types with >= --fit-min-ms in some run):")
        for name in fits["contexts"]:
            for typ, f in sorted(fits["types"].get(name, {}).items(), key=lambda kv: -(kv[1]["a"] + kv[1]["b"])):
                note = TYPE_NOTES.get(typ, "")
                L.append(f"    {name:<10} {typ:<38} a {f['a']:>7.2f}  b {f['b']:>6.2f}" + (f"  [{note}]" if note else ""))
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-context breakdown of hailortcli measure-fw-actions JSON + T(B) fit.")
    ap.add_argument("jsons", nargs="+", help="measure-fw-actions JSON files (different --batch-size, repeats ok)")
    ap.add_argument("--top-types", type=int, default=6, help="action types listed per context (0 = all, default 6)")
    ap.add_argument("--gaps", type=int, default=3, help="largest intervals listed per context (default 3)")
    ap.add_argument("--fit-min-ms", type=float, default=0.5,
                    help="fit (context, type) pairs whose time reaches this in some run (default 0.5)")
    ap.add_argument("--ref", default=f"{REF_MODEL[0]:g},{REF_MODEL[1]:g}",
                    help="reference T(B) = A + B*batch as 'A,B' ms (default 24.8,16.2)")
    ap.add_argument("--json", help="write the breakdown and fits as JSON")
    args = ap.parse_args(argv)
    try:
        ref = tuple(float(x) for x in args.ref.split(","))
        if len(ref) != 2:
            raise ValueError
    except ValueError:
        ap.error(f"--ref: expected 'A,B', got {args.ref!r}")
    if args.json and Path(args.json).resolve() in {Path(p).resolve() for p in args.jsons}:
        ap.error(f"refusing to overwrite an input file: {args.json}")

    try:
        runs = [run for p in args.jsons for run in load(p)]
    except (OSError, FwActionsError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    results = [analyze_run(r, args.gaps) for r in runs]
    sys.stdout.write("\n".join(format_run(r, args.top_types) for r in results))
    fits = fit_all(results, args.fit_min_ms)
    sys.stdout.write("\n" + format_fits(fits, results, ref))
    sys.stdout.write("note: action-type readings are inferred from names and timing, not from HailoRT documentation.\n")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"tool": "fw_actions_breakdown.py", "reference": {"a": ref[0], "b": ref[1]},
                       "runs": results, "fits": fits}, fh, indent=1)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
