"""Arrival-based latency split per run (processed frames, seq >= 30), medians in ms.

arr->dec   arrival -> decode end
dec->sub   decode end -> GPU submit (input queue wait + GPU slot wait)
arr->sub   arrival -> GPU submit
X          GPU submit -> NPU done
npu->cmp   NPU done -> complete (GPU post-processing)
arr->cmp   arrival -> complete
Usage: python stages.py <run folder>  -> prints and writes <run>/evidence/stage_split.txt
"""
import csv
import statistics as st
import sys
from pathlib import Path

out = Path(sys.argv[1])
lines = ["run                arr->dec  dec->sub  arr->sub       X  npu->cmp  arr->cmp"]
for f in sorted((out / "raw").glob("*_run*/frame_log.csv")):
    fr = [r for r in csv.DictReader(open(f)) if r["dropped"] == "0" and int(r["seq"]) >= 30 and r["arrival_ms"]]
    d = lambda a, b: st.median(float(r[b]) - float(r[a]) for r in fr)
    lines.append("%-18s %8.2f  %8.2f  %8.2f  %6.2f  %8.2f  %8.2f" % (
        f.parent.name, d("arrival_ms", "decode_end_ms"), d("decode_end_ms", "submit_ms"), d("arrival_ms", "submit_ms"),
        d("submit_ms", "npu_done_ms"), d("npu_done_ms", "complete_ms"), d("arrival_ms", "complete_ms")))
text = "\n".join(lines) + "\n"
print(text)
(out / "evidence" / "stage_split.txt").write_text(text)
