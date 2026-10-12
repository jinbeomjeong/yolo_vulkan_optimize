"""Tests for 06_tools/python/compare_detections.py (stdlib unittest, no pytest).

Run: python3 -m unittest discover -s 06_tools/python/tests

The matching rule mirrors match() in 01_src/ref/detection_match.hpp; these tests pin the Python side on tiny
synthetic detections (greedy order, tie-breaks, float32 IoU, threshold) and run the CLI on synthetic jsonl files.
"""
from __future__ import annotations

import csv
import json
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).resolve().parents[1] / "compare_detections.py"
sys.path.insert(0, str(TOOL.parent))
import compare_detections as cd  # noqa: E402


def det(cls: int, score: float, box: list[float], name: str = "obj") -> dict:
    return {"class": cls, "name": name, "score": score, "box": box}


class Float32IouTest(unittest.TestCase):
    def test_f32_rounds_to_single_precision(self) -> None:
        self.assertNotEqual(cd.f32(0.1), 0.1)
        self.assertEqual(cd.f32(0.1), struct.unpack("f", struct.pack("f", 0.1))[0])
        self.assertEqual(cd.f32(0.5), 0.5)

    def test_iou_values(self) -> None:
        self.assertEqual(cd.iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertEqual(cd.iou([0, 0, 10, 10], [20, 20, 30, 30]), 0.0)
        self.assertEqual(cd.iou([0, 0, 10, 10], [10, 0, 20, 10]), 0.0)          # touching edge
        self.assertEqual(cd.iou([0, 0, 10, 10], [5, 0, 15, 10]), cd.f32(50 / 150))
        self.assertEqual(cd.iou([0, 0, 10, 10], [0, 0, 10, 20]), 0.5)           # exactly 0.5
        self.assertEqual(cd.iou([0, 0, 0, 0], [0, 0, 0, 0]), 0.0)               # zero union

    def test_iou_is_symmetric(self) -> None:
        a, b = [1.25, 2.5, 30.75, 40.1], [3.3, 1.1, 28.9, 39.7]
        self.assertEqual(cd.iou(a, b), cd.iou(b, a))


class MatchTest(unittest.TestCase):
    def test_classes_never_match_across(self) -> None:
        self.assertEqual(cd.match([det(0, 0.9, [0, 0, 10, 10])], [det(1, 0.9, [0, 0, 10, 10])], 0.5), 0)

    def test_higher_score_reference_takes_the_test_first(self) -> None:
        low = det(0, 0.5, [0, 0, 10, 10])      # IoU 1.0 with the test box, but lower score
        high = det(0, 0.9, [1, 0, 11, 10])     # IoU 0.818
        t = det(0, 0.9, [0, 0, 10, 10])
        pairs = cd.match_pairs([low, high], [t], 0.5)
        self.assertEqual(len(pairs), 1)
        self.assertIs(pairs[0][0], high)

    def test_equal_scores_keep_original_index_order(self) -> None:
        first = det(0, 0.7, [1, 0, 11, 10])
        second = det(0, 0.7, [0, 0, 10, 10])
        pairs = cd.match_pairs([first, second], [det(0, 0.2, [0, 0, 10, 10])], 0.5)
        self.assertIs(pairs[0][0], first)

    def test_iou_tie_takes_the_earlier_test(self) -> None:
        t0, t1 = det(0, 0.1, [0, 0, 10, 10]), det(0, 0.9, [0, 0, 10, 10])
        pairs = cd.match_pairs([det(0, 0.8, [0, 0, 10, 10])], [t0, t1], 0.5)
        self.assertIs(pairs[0][1], t0)

    def test_each_test_detection_is_used_once(self) -> None:
        ref = [det(0, 0.9, [0, 0, 10, 10]), det(0, 0.8, [0, 0, 10, 10])]
        self.assertEqual(cd.match(ref, [det(0, 0.9, [0, 0, 10, 10])], 0.5), 1)

    def test_threshold_is_inclusive(self) -> None:
        ref, test = [det(0, 0.9, [0, 0, 10, 10])], [det(0, 0.9, [0, 0, 10, 20])]   # IoU exactly 0.5
        self.assertEqual(cd.match(ref, test, 0.5), 1)
        self.assertEqual(cd.match(ref, test, 0.7), 0)

    def test_empty_inputs(self) -> None:
        self.assertEqual(cd.match_pairs([], [det(0, 0.9, [0, 0, 1, 1])], 0.5), [])
        self.assertEqual(cd.match_pairs([det(0, 0.9, [0, 0, 1, 1])], [], 0.5), [])


class StatsTest(unittest.TestCase):
    def test_coord_err(self) -> None:
        self.assertEqual(cd.coord_err([0, 0, 10, 10], [1, -2, 10, 13]), 3)

    def test_percentile_linear_interpolation(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(cd.percentile(vals, 0.0), 1.0)
        self.assertEqual(cd.percentile(vals, 0.5), 2.5)
        self.assertAlmostEqual(cd.percentile(vals, 0.9), 3.7)
        self.assertEqual(cd.percentile(vals, 1.0), 4.0)
        self.assertTrue(math.isnan(cd.percentile([], 0.5)))

    def test_err_stats(self) -> None:
        s = cd.err_stats([3.0, 1.0, 2.0])
        self.assertEqual((s["n"], s["max"], s["mean"], s["p50"]), (3, 3.0, 2.0, 2.0))
        self.assertAlmostEqual(s["p90"], 2.8)
        empty = cd.err_stats([])
        self.assertEqual(empty["n"], 0)
        self.assertTrue(all(math.isnan(empty[k]) for k in ("max", "mean", "p50", "p90")))
        self.assertEqual((cd.fmt_err(empty["max"]), cd.csv_err(empty["max"])), ("n/a", ""))
        self.assertEqual(cd.fmt_err(0.5, 2), "0.50")


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def jsonl(self, name: str, frames: dict[int, list[dict]]) -> Path:
        p = self.dir / name
        lines = [json.dumps({"frame": f, "detections": d}) for f, d in frames.items()]
        p.write_text("\n".join(lines) + "\n\n", encoding="utf-8")   # blank line is skipped
        return p

    def run_tool(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run([sys.executable, str(TOOL), *args], capture_output=True, text=True, timeout=30)

    def test_report_and_csv(self) -> None:
        ref = self.jsonl("ref.jsonl", {
            0: [det(0, 0.9, [0, 0, 10, 10], "person"), det(2, 0.8, [20, 20, 40, 40], "car")],
            1: [det(0, 0.7, [0, 0, 10, 10], "person")],
            2: [det(0, 0.7, [0, 0, 10, 10], "person")],      # not in test: ignored
        })
        test = self.jsonl("test.jsonl", {
            0: [det(0, 0.95, [0, 0, 10, 11], "person"), det(2, 0.8, [20, 20, 40, 40], "car")],
            1: [],
            3: [det(0, 0.7, [0, 0, 10, 10], "person")],      # not in ref: ignored
        })
        out_csv = self.dir / "out.csv"
        r = self.run_tool(str(ref), str(test), "--csv", str(out_csv), "--label", "L1")
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = r.stdout.splitlines()
        self.assertIn("[warn] frame counts differ: ref 3, test 3, common 2", lines)
        self.assertIn("frames           : 2", lines)
        self.assertIn("detections test  : 2  (-33.33 %)", lines)
        # person IoU 100/110 = 0.909 passes every threshold; denom = max(2,2) + max(1,0) = 3
        for thr in ("0.5", "0.7", "0.9"):
            self.assertIn(f"match @ IoU {thr} : 2 / 3 = 0.6667", lines)

        with out_csv.open(encoding="utf-8", newline="") as f:
            rows = list(csv.reader(f))
        head = dict(zip(rows[0], rows[1]))
        self.assertEqual(head["label"], "L1")
        self.assertEqual((head["det_ref"], head["det_test"], head["det_diff_pct"]), ("3", "2", "-33.3333"))
        self.assertEqual((head["match_ratio_iou9"], head["denom"]), ("0.6667", "3"))
        # coordinate errors of the two pairs: 1 (person y2) and 0 (car)
        self.assertEqual((head["max_coord_err_px"], head["mean_coord_err_px"], head["p90_coord_err_px"]),
                         ("1.000000", "0.500000", "0.900000"))
        self.assertEqual((head["coord_err_pairs"], head["coord_err_iou"]), ("2", "0.9"))
        self.assertEqual(rows[2], [])
        self.assertEqual(rows[3], ["class_id", "class_name", "ref", "test", "diff_pct"])
        self.assertEqual(rows[4:], [["0", "person", "2", "1", "-50.00"], ["2", "car", "1", "1", "0.00"]])

    def test_no_common_frames_fails(self) -> None:
        ref = self.jsonl("a.jsonl", {0: []})
        test = self.jsonl("b.jsonl", {1: []})
        r = self.run_tool(str(ref), str(test))
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("no common frames", r.stderr)

    def test_zero_matches_prints_na(self) -> None:
        ref = self.jsonl("a.jsonl", {0: [det(0, 0.9, [0, 0, 10, 10])]})
        test = self.jsonl("b.jsonl", {0: [det(1, 0.9, [0, 0, 10, 10])]})
        r = self.run_tool(str(ref), str(test))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("match @ IoU 0.5 : 0 / 1 = 0.0000", r.stdout)
        self.assertIn("0.9        0          n/a          n/a          n/a          n/a", r.stdout)


if __name__ == "__main__":
    unittest.main()
