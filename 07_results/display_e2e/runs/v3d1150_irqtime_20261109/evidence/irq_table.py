"""Per-config IRQ table of a display_bench IRQ_TRACE=1 run folder: per run, per-frame (900 frames) hardirq count,
hardirq ms (idle part), softirq ms (idle part), mean us per hardirq, and the uncounted share
(idle hardirq + softirq, `uncounted_ms_per_frame`); then medians [min-max] over runs.
Checked: on display_e2e/runs/v3d1150_irqtime_20261016 this reproduces its summary table (0.211 / 0.222 / 0.328 / 0.399).
Usage: python irq_table.py <irq run folder> [frames=900]
"""
import csv
import statistics as st
import sys

run = sys.argv[1]
F = float(sys.argv[2]) if len(sys.argv) > 2 else 900.0
rows = list(csv.DictReader(open(f"{run}/results_raw_irq.csv")))
for cfg in ["reg_host_off", "disp_off", "disp_asap", "disp_source"]:
    g = [r for r in rows if r["label"] == cfg]
    if not g:
        continue
    per = lambda k: [float(r[k]) / F for r in g]
    med = lambda v: st.median(v)
    us = [1000 * float(r["hardirq_ms"]) / float(r["hardirq_count"]) for r in g]
    u = [float(r["uncounted_ms_per_frame"]) for r in g]
    print(f"{cfg:13s} hardirq/f {med(per('hardirq_count')):.1f}  hardirq {med(per('hardirq_ms')):.3f} "
          f"(idle {med(per('hardirq_idle_ms')):.3f})  softirq {med(per('softirq_ms')):.3f} (idle {med(per('softirq_idle_ms')):.3f})"
          f"  us/irq {med(us):.2f}  uncounted {med(u):.3f} [{min(u):.3f}-{max(u):.3f}]  unmatched {max(int(r['unmatched']) for r in g)}  n={len(g)}")
