#!/usr/bin/env python3
"""두 파이프라인의 detections.jsonl을 비교한다.

매칭 규칙은 05_tests/test_detections.cpp의 iou()/match()와 같다:
  같은 클래스만, 참조 검출을 점수 내림차순으로 훑으며 IoU가 가장 큰 미사용 테스트 검출과 greedy 매칭.
  매칭 비율 = matched / max(ref_count, test_count)  (프레임 합계)

좌표 오차(G4)는 매칭된 (ref, test) 쌍마다
  err = max(|dx1|, |dy1|, |dx2|, |dy2|)   [원본 프레임 픽셀 좌표계]
를 모아 max/mean/p50/p90을 IoU 임계값별로 낸다.
백분위는 01_src/pipeline/main.cpp의 stats()와 같은 선형 보간이다.
CSV에서 접미사 없는 *_coord_err_px 열은 PRIMARY_IOU 기준이다.

사용법:
  python 08_python/compare_detections.py REF.jsonl TEST.jsonl [--csv OUT.csv] [--label NAME]
"""
import argparse
import csv
import json
import sys
from collections import Counter

IOUS = (0.5, 0.7, 0.9)
# 접미사 없는 좌표 오차 열의 기준. test_detections.cpp의 kIouThreshold와 같다.
PRIMARY_IOU = 0.9


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0.0 else 0.0


def match_pairs(ref, test, thr):
    """test_detections.cpp의 match()와 같은 greedy 매칭. 매칭된 (ref, test) 쌍 목록을 반환."""
    used = [False] * len(test)
    pairs = []
    for r in sorted(ref, key=lambda d: -d["score"]):
        best, best_j = 0.0, -1
        for j, t in enumerate(test):
            if used[j] or t["class"] != r["class"]:
                continue
            v = iou(r["box"], t["box"])
            if v > best:
                best, best_j = v, j
        if best_j >= 0 and best >= thr:
            used[best_j] = True
            pairs.append((r, test[best_j]))
    return pairs


def match(ref, test, thr):
    """매칭 수를 반환. match_pairs()와 같은 규칙이다."""
    return len(match_pairs(ref, test, thr))


def coord_err(a, b):
    """매칭 쌍의 네 좌표 절대 오차 중 최댓값 [원본 프레임 픽셀]."""
    return max(abs(a[k] - b[k]) for k in range(4))


def percentile(sorted_vals, p):
    """main.cpp stats()와 같은 선형 보간 백분위. 빈 목록이면 NaN."""
    if not sorted_vals:
        return float("nan")
    idx = p * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (idx - lo)


def err_stats(errs):
    """좌표 오차 목록 → {n, max, mean, p50, p90}. 매칭 쌍이 0개면 n=0, 나머지는 NaN."""
    n = len(errs)
    if n == 0:
        nan = float("nan")
        return {"n": 0, "max": nan, "mean": nan, "p50": nan, "p90": nan}
    s = sorted(errs)
    return {"n": n, "max": s[-1], "mean": sum(s) / n, "p50": percentile(s, 0.5), "p90": percentile(s, 0.9)}


def fmt_err(v, prec=6):
    """NaN(매칭 쌍 0개)이면 n/a."""
    return "n/a" if v != v else f"{v:.{prec}f}"


def csv_err(v, prec=6):
    """CSV용. NaN이면 빈 칸."""
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
        sys.exit("공통 프레임이 없다")
    if len(common) != len(ref) or len(common) != len(test):
        print(f"[warn] 프레임 수가 다르다: ref {len(ref)}, test {len(test)}, 공통 {len(common)}")

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
          f"detections test  : {n_test}  (n/a — ref 검출 0)")
    for thr in IOUS:
        ratio = matched[thr] / denom if denom else 0.0
        print(f"match @ IoU {thr:.1f} : {matched[thr]} / {denom} = {ratio:.4f}")

    print("\ncoord err px (매칭 쌍의 max(|dx1|,|dy1|,|dx2|,|dy2|))")
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
