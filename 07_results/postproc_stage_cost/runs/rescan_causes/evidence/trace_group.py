#!/usr/bin/env python3
"""Replay yolo_decode.comp stage ① (pmax4, ge_cnt) bit for bit for one invocation (4 anchors) of a raw fixture.

Picks the first invocation of 04_data/raw_tensors frame 0, P3 matching each case and prints every class word that
matters, the ge_cnt increment, pmax4 after each class, and the ⑤ path per anchor. Usage: python trace_group.py
"""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from rescan_causes import COCO  # noqa: E402
M = 0xffffffff
def pack4(w, c):
    return [((w << 7) & 0x7f80) | c, ((w >> 1) & 0x7f80) | c, ((w >> 9) & 0x7f80) | c, ((w >> 17) & 0x7f80) | c]

def load(fdir, frame, scale_id):
    meta = json.loads((Path(fdir) / "fixtures.json").read_text())
    s = next(x for x in meta["streams"] if x["role"] == "cls" and x["scale_id"] == scale_id)
    raw = np.fromfile(Path(fdir) / f"frame{frame:03d}_{s['file']}.bin", dtype=np.uint8)
    return s, raw.view("<u4").reshape(s["grid_h"], 80, s["w_padded"] // 4)   # [h][c][word]: word = 4 anchors w=4wi..4wi+3

def find(fdir, frame, scale_id, thr, want):
    s, words = load(fdir, frame, scale_id)
    q_thr_low = math.floor(thr / s["qp_scale"]); t_sw = min(q_thr_low, 128)
    for h in range(s["grid_h"]):
        for wi in range(s["grid_w"] // 4):
            q = np.stack([(words[h, :, wi] >> (8 * a)) & 0xff for a in range(4)])   # (anchor, class)
            n = (q >= t_sw).sum(1); mx = q.max(1)
            if want(n, mx, q_thr_low): return s, words, h, wi, q_thr_low, t_sw

def trace(fdir, frame, scale_id, thr, want, title):
    s, words, h, wi, q_thr_low, t_sw = find(fdir, frame, scale_id, thr, want)
    print(f"\n=== {title}: {Path(fdir).name} frame {frame}, P{scale_id+3}, thr {thr}, invocation (wi={wi}, h={h}) = anchors w={4*wi}..{4*wi+3}")
    print(f"    cls_scale={s['qp_scale']:.8f}  q_thr_low=floor({thr}/scale)={q_thr_low}  t_sw={t_sw}  t_sw4={t_sw*0x01010101:#010x}")
    pmax4 = [0, 0, 0, 0]; ge = 0; t4 = t_sw * 0x01010101
    print("    c   class           word(hex)    bytes a0..a3        ge_cnt increment  pmax4 after (q<<7|c, hex)")
    for c in range(80):
        w = int(words[h, c, wi])
        pmax4 = [max(p, k) for p, k in zip(pmax4, pack4(w, c))]
        inc = ((((w | 0x80808080) - t4) & M | w) & 0x80808080) >> 7
        ge = (ge + inc) & M
        b = [(w >> (8 * a)) & 0xff for a in range(4)]
        if max(b) >= q_thr_low * 0.5 or inc:
            print(f"   {c:2d}  {COCO[c]:14s}  {w:#010x}   {b[0]:3d} {b[1]:3d} {b[2]:3d} {b[3]:3d}     {inc:#010x}       "
                  + " ".join(f"{p:#06x}" for p in pmax4))
    print(f"    final pmax4 = [{', '.join(f'{p:#06x}' for p in pmax4)}]   ge_cnt = {ge:#010x}")
    thr_pk = min(q_thr_low, 256) << 7
    print(f"    ② pmax = {max(pmax4):#06x} vs thr_pk = {q_thr_low}<<7 = {thr_pk:#06x} -> {'continue' if max(pmax4) >= thr_pk else 'return (group rejected)'}")
    for a in range(4):
        p = pmax4[a]; qm, c = p >> 7, p & 0x7f; n = (ge >> (8 * a)) & 0xff
        score = (np.float32(qm) - np.float32(s["qp_zp"])) * np.float32(s["qp_scale"])
        if p < thr_pk: path = "skip (no class reaches q_thr_low)"
        elif n == 1: path = f"SHORTCUT -> score {score:.4f} {'>=' if score >= np.float32(thr) else '<'} {thr} -> " + ("emit" if score >= np.float32(thr) else "no emit")
        else: path = f"RESCAN 80 words ({n} loose passes)"
        print(f"    anchor a={a} (w={4*wi+a}): pmax4={p:#06x} -> q_max={qm} class={c}({COCO[c]})  ge_cnt byte={n}  => {path}")

root = Path(__file__).resolve().parents[5]   # evidence -> rescan_causes -> runs -> postproc_stage_cost -> 07_results -> repo
trace(f"{root}/04_data/raw_tensors", 0, 0, 0.2, lambda n, mx, T: (mx >= T).sum() == 1 and (n == 0).sum() >= 2, "case A (operating point)")
trace(f"{root}/04_data/raw_tensors", 0, 0, 0.02, lambda n, mx, T: (n >= 2).any() and (n == 1).any(), "case B (low threshold, rescan)")
