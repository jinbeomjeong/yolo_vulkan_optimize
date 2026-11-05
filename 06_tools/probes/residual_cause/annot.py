#!/usr/bin/env python3
"""Cycle samples per instruction of a loop, from perf script symbol offsets (samples have skid)."""
import collections, re, subprocess, sys
data, pat, fstart, lo, hi, binf = sys.argv[1], sys.argv[2], int(sys.argv[3],16), int(sys.argv[4],16), int(sys.argv[5],16), sys.argv[6]
out = subprocess.run(["perf","script","-i",data,"--no-demangle","-F","ip,sym,symoff"],capture_output=True,text=True).stdout
cnt = collections.Counter(); tot = 0
for l in out.splitlines():
    m = re.search(r"\s(\S+)\+0x([0-9a-f]+)", l)
    if not m: continue
    tot += 1
    if re.search(pat, m[1]): cnt[fstart+int(m[2],16)] += 1
dis = subprocess.run(["objdump","-d","--no-show-raw-insn",f"--start-address={lo:#x}",f"--stop-address={hi+4:#x}",binf],capture_output=True,text=True).stdout
rows = [(cnt[int(m[1],16)],int(m[1],16),m[2],m[3].split("<")[0].split("//")[0].strip()) for m in re.finditer(r"^\s+([0-9a-f]+):\t(\S+)\s*(.*)$",dis,re.M)]
s = sum(r[0] for r in rows)
print(f"{pat}: loop samples {s} of {sum(cnt.values())} in function, {len(rows)} instructions")
by = collections.Counter(); n = collections.Counter()
for c,a,mn,op in rows:
    k = "ldr_b" if mn=="ldr" and op.startswith("b") else mn
    by[k]+=c; n[k]+=1
print(" by mnemonic (%/count):", ", ".join(f"{k} {100*by[k]/s:.1f}/{n[k]}" for k,_ in by.most_common(14)))
print(" top:", "; ".join(f"{100*c/s:.1f}% {a:x} {mn} {op}" for c,a,mn,op in sorted(rows,reverse=True)[:8]))
