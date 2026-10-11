#!/usr/bin/env python3
"""Count the anchors that take the yolo_decode.comp ⑤ rescan fallback and classify why (07_results/postproc_stage_cost).

Offline, from raw NPU output fixtures (no GPU, no NPU). Replays the shader's decisions exactly:
  - an anchor reaches ⑤ when its max q >= q_thr_low (pmax4 >= thr_pk), q_thr_low = floor(thr / cls_scale)
  - shortcut when the loose count #{c : q_c >= t_sw} == 1, t_sw = min(q_thr_low, 128) (the ge_cnt byte)
  - otherwise rescan; the definitive test is the float32 (q - zp) * scale >= thr
A rescan anchor is classified by how many classes pass the definitive test:
  >= 2  -> real multi-class. Split by class pair: both in one similar-class group = cause 1 (similar classes),
           else cause 2 (overlapping objects). This split is a heuristic: the tensors cannot tell two objects apart.
  == 1  -> cause 3 (quantization false alarm of the loose bound); == 0 -> cause 3 with nothing emitted.
Usage: python rescan_causes.py <fixture_dir>...   (fixture_dir holds fixtures.json + frameNNN_<stream>.bin)
"""
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign parking_meter "
        "bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag tie suitcase frisbee skis "
        "snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard tennis_racket bottle wine_glass cup fork "
        "knife spoon bowl banana apple sandwich orange broccoli carrot hot_dog pizza donut cake chair couch potted_plant bed "
        "dining_table toilet tv laptop mouse remote keyboard cell_phone microwave oven toaster sink refrigerator book clock "
        "vase scissors teddy_bear hair_drier toothbrush").split()
SIMILAR = ({"car", "truck", "bus"}, {"handbag", "backpack", "suitcase"})
THRESHOLDS = (0.2, 0.05, 0.02)


def run(fdir, thr):
    meta = json.loads((Path(fdir) / "fixtures.json").read_text())
    nf, ncls = meta["frames"], meta["num_classes"]
    thr32 = np.float32(thr)
    tot, pairs = Counter(), Counter()
    for s in (x for x in meta["streams"] if x["role"] == "cls"):
        H, W, Wp = s["grid_h"], s["grid_w"], s["w_padded"]
        scale, zp = np.float32(s["qp_scale"]), np.float32(s["qp_zp"])
        q_thr_low = math.floor(thr / s["qp_scale"])
        t_sw = min(q_thr_low, 128)
        for f in range(nf):
            raw = np.fromfile(Path(fdir) / f"frame{f:03d}_{s['file']}.bin", dtype=np.uint8)
            q = raw.reshape(H, ncls, Wp)[:, :, :W].transpose(0, 2, 1).reshape(-1, ncls)   # NHCW -> (anchor, class)
            n_pre = (q >= t_sw).sum(1)
            score = (q.astype(np.float32) - zp) * scale
            n_def = (score >= thr32).sum(1)
            for a in np.nonzero(q.max(1) >= q_thr_low)[0]:
                tot["enter"] += 1
                if n_pre[a] == 1:
                    tot["shortcut"] += 1
                    tot["shortcut_emit" if n_def[a] == 1 else "shortcut_noemit"] += 1
                    continue
                tot["rescan"] += 1
                if n_def[a] >= 2:
                    cs = sorted(np.nonzero(score[a] >= thr32)[0], key=lambda c: -int(q[a, c]))
                    a_name, b_name = COCO[cs[0]], COCO[cs[1]]
                    pairs[(a_name, b_name)] += 1
                    tot["c1_similar" if any(a_name in g and b_name in g for g in SIMILAR) else "c2_overlap"] += 1
                else:
                    tot["c3_loose_one" if n_def[a] == 1 else "c3_loose_none"] += 1
    return tot, pairs


def main():
    print("input        thr    reach5  shortcut  rescan  rescan%  c1_similar  c2_overlap  c3_loose(1/0)  emits(shortcut)  no-emit(shortcut)")
    for fdir in sys.argv[1:]:
        for thr in THRESHOLDS:
            t, p = run(fdir, thr)
            r = t["rescan"]
            print(f"{Path(fdir).name:12s} {thr:<5}  {t['enter']:6d}  {t['shortcut']:8d}  {r:6d}  {100 * r / max(t['enter'], 1):6.2f}%"
                  f"  {t['c1_similar']:10d}  {t['c2_overlap']:10d}  {t['c3_loose_one']:6d}/{t['c3_loose_none']:<6d}"
                  f"  {t['shortcut_emit']:15d}  {t['shortcut_noemit']:17d}")
            if p:
                print("    pairs (top class + 2nd):", ", ".join(f"{a}+{b} {n}" for (a, b), n in p.most_common(10)))


if __name__ == "__main__":
    main()
