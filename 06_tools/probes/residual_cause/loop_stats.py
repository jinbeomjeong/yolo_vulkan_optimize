#!/usr/bin/env python3
"""Static stats of the pixel loops of a function: smallest cycle through the color-store block.

Usage: loop_stats.py <binary> <symbol-regex> ...
For each loop found (backward branch whose range contains an instruction of interest) prints the instruction count,
mnemonic histogram, stack accesses and conditional branches. The range is the static span, so cold branches that
jump out of it are not subtracted.
"""
import collections
import re
import subprocess
import sys

binary = sys.argv[1]
nm = subprocess.run(["nm", "-S", binary], capture_output=True, text=True).stdout
for pat in sys.argv[2:]:
    for line in nm.splitlines():
        f = line.split()
        if len(f) != 4 or not re.search(pat, f[3]):
            continue
        a, s = int(f[0], 16), int(f[1], 16)
        dis = subprocess.run(["objdump", "-d", "--no-show-raw-insn", f"--start-address={a:#x}",
                              f"--stop-address={a+s:#x}", binary], capture_output=True, text=True).stdout
        ins = []
        for m in re.finditer(r"^\s+([0-9a-f]+):\t(\S+)\s*(.*)$", dis, re.M):
            ins.append((int(m[1], 16), m[2], m[3]))
        addr = {i[0]: k for k, i in enumerate(ins)}
        print(f"== {f[3]} size={s}")
        total_stack = [i for i in ins if "[sp" in i[2]]
        print(f"   stack refs in whole function: {len(total_stack)}")
        loops = []
        for k, (ad, mn, op) in enumerate(ins):
            if mn.startswith("b") and mn not in ("bl", "blr", "br", "bfi", "bic", "bfxil"):
                t = re.match(r"(?:[wx]\d+, (?:#\S+, )?)?([0-9a-f]+)", op)
                if t and int(t[1], 16) in addr and int(t[1], 16) <= ad:
                    loops.append((addr[int(t[1], 16)], k))
        for b, e in sorted(set(loops)):
            body = ins[b : e + 1]
            if not any(i[1] in ("fmadd", "fcvtzu") for i in body):
                continue
            h = collections.Counter(i[1] for i in body)
            st = sum("[sp" in i[2] for i in body)
            lb = sum(1 for i in body if i[1].startswith("ldr") and re.search(r"\[x\d+(, #\d+)?\]$", i[2]))
            br = sum(1 for i in body if i[1].startswith("b.") or i[1] in ("b", "cbz", "cbnz", "tbz", "tbnz"))
            print(f"   loop {ins[b][0]:x}..{ins[e][0]:x} n={len(body)} stack={st} plain_load={lb} br={br} "
                  f"scvtf={h['scvtf']} fcvtzs={h['fcvtzs']} madd={h['madd']} ldr_b={sum(1 for i in body if i[1]=='ldr' and i[2].startswith('b'))} "
                  f"fcsel={h['fcsel']} fcmpe={h['fcmpe']} csel={h['csel']} simd={sum(1 for i in body if re.search(r'v[0-9]+\.', i[2]))}")
