#!/usr/bin/env python3
"""Compare detections.jsonl of two pipelines.

The matching rule is **equivalent** to match() in 01_src/ref/detection_match.hpp (the C++ side is canonical):
  same class only; walk reference detections in the total order (score descending, original index ascending)
  and greedily match the unused test detection with the largest IoU (on IoU ties, the earlier one).
  match ratio = matched / max(ref_count, test_count)  (summed over frames)

Two things are aligned with C++ for equivalence (before 2026-09-21 both diverged):
  · Sorting: C++ used std::sort (unstable) and this used sorted (stable), so the choice diverged on score ties.
    Now both sides use the (-score, index) total order.
  · Precision: C++ computes IoU in float32. Here f32() rounds after every operation.
    (Doing one basic operation in double and rounding to float32 gives exactly the float32 result.
     g++ does not contract to FMA under -std=c++20 (ISO mode), so this emulation holds.)

Coordinate error (G4): for each matched (ref, test) pair
  err = max(|dx1|, |dy1|, |dx2|, |dy2|)   [original frame pixel coordinates]
is collected and max/mean/p50/p90 are reported per IoU threshold.
Percentiles use the same linear interpolation as 01_src/toolkit/stats.hpp.
In the CSV, the unsuffixed *_coord_err_px columns are based on PRIMARY_IOU.

Usage:
  python 06_tools/python/compare_detections.py REF.jsonl TEST.jsonl [--csv OUT.csv] [--label NAME]
"""
import argparse
import csv
import json
import struct
import sys
from collections import Counter

IOUS = (0.5, 0.7, 0.9)
# Reference for the unsuffixed coordinate error columns. Same as kIouThreshold in ref/detection_match.hpp.
PRIMARY_IOU = 0.9


def f32(x):
    """Round to float32. Used to emulate C++ float arithmetic."""
    return struct.unpack("f", struct.pack("f", x))[0]


def iou(a, b):
    """Emulates iou_xyxy() of ref/cpu_yolov8_decode.hpp in float32 (same operation order)."""
    a = [f32(v) for v in a]
    b = [f32(v) for v in b]
    ix = max(0.0, f32(min(a[2], b[2]) - max(a[0], b[0])))
    iy = max(0.0, f32(min(a[3], b[3]) - max(a[1], b[1])))
    inter = f32(ix * iy)
    area_a = f32(f32(a[2] - a[0]) * f32(a[3] - a[1]))
    area_b = f32(f32(b[2] - b[0]) * f32(b[3] - b[1]))
    ua = f32(f32(area_a + area_b) - inter)
    return f32(inter / ua) if ua > 0.0 else 0.0


def match_pairs(ref, test, thr):
    """Greedy matching equivalent to match() in ref/detection_match.hpp. Returns the list of matched (ref, test) pairs."""
    used = [False] * len(test)
    pairs = []
    thr32 = f32(thr)
    order = sorted(range(len(ref)), key=lambda i: (-f32(ref[i]["score"]), i))
    for i in order:
        r = ref[i]
        best, best_j = 0.0, -1
        for j, t in enumerate(test):
            if used[j] or t["class"] != r["class"]:
                continue
            v = iou(r["box"], t["box"])
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= thr32:
            used[best_j] = True
            pairs.append((r, test[best_j]))
    return pairs


def match(ref, test, thr):
    """Return the match count. Same rule as match_pairs()."""
    return len(match_pairs(ref, test, thr))


def coord_err(a, b):
    """Maximum of the four absolute coordinate errors of a matched pair [original frame pixels]."""
    return max(abs(a[k] - b[k]) for k in range(4))


def percentile(sorted_vals, p):
    """Linear-interpolation percentile, same as main.cpp stats(). NaN for an empty list."""
    if not sorted_vals:
        return float("nan")
    idx = p * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def err_stats(errs):
    """Coordinate error list -> {n, max, mean, p50, p90}. With 0 matched pairs, n=0 and the rest NaN."""
    n = len(errs)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "max": nan, "mean": nan, "p50": nan, "p90": nan}
    s = sorted(errs)
    return {"n": n, "max": s[-1], "mean": sum(s) / n, "p50": percentile(s, 0.5), "p90": percentile(s, 0.9)}


def fmt_err(v, prec=6):
    """n/a if NaN (0 matched pairs)."""
    return "n/a" if v != v else f"{v:.{prec}f}"


def csv_err(v, prec=6):
    """For CSV. Empty cell if NaN."""
    return "" if v != v else f"{v:.{prec}f}"


def load(path):
    frames = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            frames[o["frame"]] = o["detections"]
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--csv")
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    ref, test = load(a.ref), load(a.test)
    common = sorted(set(ref) & set(test))
    if not common:
        sys.exit("no common frames")
    if len(common) != len(ref) or len(common) != len(test):
        print(f"[warn] frame counts differ: ref {len(ref)}, test {len(test)}, common {len(common)}")

    n_ref = sum(len(ref[i]) for i in common)
    n_test = sum(len(test[i]) for i in common)
    matched = {t: 0 for t in IOUS}
    errs = {t: [] for t in IOUS}
    denom = 0
    for i in common:
        r, t = ref[i], test[i]
        denom += max(len(r), len(t))
        for thr in IOUS:
            pairs = match_pairs(r, t, thr)
            matched[thr] += len(pairs)
            errs[thr].extend(coord_err(p[0]["box"], p[1]["box"]) for p in pairs)
    stats = {thr: err_stats(errs[thr]) for thr in IOUS}

    cls_ref = Counter(d["class"] for i in common for d in ref[i])
    cls_test = Counter(d["class"] for i in common for d in test[i])
    name = {d["class"]: d["name"] for i in common for d in ref[i]}
    name.update({d["class"]: d["name"] for i in common for d in test[i]})

    print(f"label            : {a.label or '(none)'}")
    print(f"frames           : {len(common)}")
    print(f"detections ref   : {n_ref}")
    pct = (100.0 * (n_test - n_ref) / n_ref) if n_ref else float("nan")
    print(f"detections test  : {n_test}  ({pct:+.2f} %)" if n_ref else
          f"detections test  : {n_test}  (n/a — 0 ref detections)")
    for thr in IOUS:
        ratio = matched[thr] / denom if denom else 0.0
        print(f"match @ IoU {thr:.1f} : {matched[thr]} / {denom} = {ratio:.4f}")

    print("\ncoord err px (max(|dx1|,|dy1|,|dx2|,|dy2|) of matched pairs)")
    print("IoU    pairs          max         mean          p50          p90")
    for thr in IOUS:
        s = stats[thr]
        print(f"{thr:.1f}  {s['n']:>7}  {fmt_err(s['max']):>11}  {fmt_err(s['mean']):>11}  "
              f"{fmt_err(s['p50']):>11}  {fmt_err(s['p90']):>11}")

    print("\nclass                 ref   test    diff")
    rows = []
    for c in sorted(set(cls_ref) | set(cls_test), key=lambda c: -cls_ref.get(c, 0)):
        r, t = cls_ref.get(c, 0), cls_test.get(c, 0)
        pct = 100.0 * (t - r) / r if r else float("nan")
        rows.append((c, name.get(c, str(c)), r, t, pct))
    for c, nm, r, t, pct in rows:
        print(f"{nm:<20} {r:>5}  {t:>5}  {pct:+7.2f} %")

    if a.csv:
        prim = stats[PRIMARY_IOU]
        with open(a.csv, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["label", "frames", "det_ref", "det_test", "det_diff_pct"] +
                       [f"match_ratio_iou{int(t * 10)}" for t in IOUS] + ["denom"] +
                       ["max_coord_err_px", "mean_coord_err_px", "p50_coord_err_px", "p90_coord_err_px",
                        "coord_err_pairs", "coord_err_iou"] +
                       [f"{k}_coord_err_px_iou{int(t * 10)}" for t in IOUS
                        for k in ("max", "mean", "p50", "p90")])
            w.writerow([a.label, len(common), n_ref, n_test,
                        f"{100.0 * (n_test - n_ref) / n_ref:.4f}" if n_ref else ""] +
                       [f"{matched[t] / denom:.4f}" for t in IOUS] + [denom] +
                       [csv_err(prim["max"]), csv_err(prim["mean"]), csv_err(prim["p50"]), csv_err(prim["p90"]),
                        prim["n"], f"{PRIMARY_IOU:.1f}"] +
                       [csv_err(stats[t][k]) for t in IOUS for k in ("max", "mean", "p50", "p90")])
            w.writerow([])
            w.writerow(["class_id", "class_name", "ref", "test", "diff_pct"])
            for c, nm, r, t, pct in rows:
                w.writerow([c, nm, r, t, f"{pct:.2f}"])
        print(f"\nsaved: {a.csv}")


if __name__ == "__main__":
    main()
