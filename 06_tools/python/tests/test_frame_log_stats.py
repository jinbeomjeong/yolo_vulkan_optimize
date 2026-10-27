"""Tests for 06_tools/python/frame_log_stats.py (stdlib unittest, no pytest).

Run: python3 -m unittest discover -s 06_tools/python/tests

All inputs are small synthetic --frame-log CSVs written to a temp dir; no real run logs are read.
"""
from __future__ import annotations

import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import frame_log_stats as fls  # noqa: E402

HEADER = ",".join(fls.HEADER)


def row(src, seq, dropped, arr, dec, sub=None, npu=None, post=None, done=None) -> str:
    def f(v):
        return "" if v is None else f"{v:.3f}"
    return ",".join([str(src), "" if seq is None else str(seq), str(dropped),
                     f(arr), f(dec), f(sub), f(npu), f(post), f(done)])


# Paced batch-2 run, 8 arrivals at 33.333 ms:
#   src 0,1 -> batch A (npu_done 100, 102)   src 2 dropped   src 3,4 -> batch B (npu_done 200, 202)
#   src 5 alone (npu_done 300)   src 6 dropped   src 7 abandoned
PACED_B2 = [
    row(0, 0, 0, 0.0, 5.0, 5.0, 100.0, 103.0, 103.0),
    row(1, 1, 0, 33.333, 38.0, 38.0, 102.0, 104.0, 104.5),
    row(2, None, 1, 66.667, 70.0),
    row(3, 2, 0, 100.0, 104.0, 110.0, 200.0, 203.0, 203.5),
    row(4, 3, 0, 133.333, 137.0, 140.0, 202.0, 205.0, 205.0),
    row(5, 4, 0, 166.667, 170.0, 171.0, 300.0, 302.0, 302.0),
    row(6, None, 1, 200.0, 204.0),
    row(7, None, 2, 233.333, 237.0),
]

HEADER10 = HEADER + "," + fls.PRESENT_FIELD


def with_present(r: str, present: float | None) -> str:
    """Append the 10th column (present_ms, --display on) to a 9-column row."""
    return r + "," + ("" if present is None else f"{present:.3f}")


# PACED_B2 with --display on: src 4 processed but not presented; present diffs 16.667, 83.333, 100
#   display (present - complete): src 0 7.0, src 1 22.167, src 3 6.5, src 5 8.0
PACED_B2_DISP = [with_present(r, p) for r, p in
                 zip(PACED_B2, [110.0, 126.667, None, 210.0, None, 310.0, None, None])]


class Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def write(self, name: str, rows: list[str], header: str = HEADER) -> Path:
        p = self.dir / name
        p.write_text(header + "\n" + "".join(r + "\n" for r in rows), encoding="utf-8")
        return p


class ParseTest(Tmp):
    def test_empty_and_nan_fields_are_none(self) -> None:
        p = self.write("a.csv", ["0,0,0,,1.5,2.0,nan,,9.0", "1,,1,3.0,4.0,,,,"])
        a, b = fls.read_frame_log(p)
        self.assertIsNone(a.arrival)
        self.assertIsNone(a.npu_done)
        self.assertIsNone(a.gpu_post_done)
        self.assertEqual((a.decode_end, a.submit, a.complete), (1.5, 2.0, 9.0))
        self.assertIsNone(b.seq)
        self.assertEqual(b.dropped, 1)

    def test_missing_column_is_rejected(self) -> None:
        p = self.write("bad.csv", ["0,0,0"], header="src,seq,dropped")
        with self.assertRaises(fls.FrameLogError):
            fls.read_frame_log(p)

    def test_bad_number_reports_line(self) -> None:
        p = self.write("bad.csv", ["0,0,0,,x,,,,"])
        with self.assertRaisesRegex(fls.FrameLogError, r"bad\.csv:2"):
            fls.read_frame_log(p)

    def test_present_column(self) -> None:
        p = self.write("d.csv", ["0,0,0,,1.0,2.0,3.0,,4.0,20.5", "1,1,0,,5.0,6.0,7.0,,8.0,", "2,,1,9.0,10.0,,,,,"],
                       header=HEADER10)
        frames, has_present = fls.parse_frame_log(p)
        self.assertTrue(has_present)
        self.assertEqual([f.present for f in frames], [20.5, None, None])

    def test_nine_columns_have_no_present(self) -> None:
        p = self.write("a.csv", ["0,0,0,,1.5,2.0,3.0,,9.0"])
        frames, has_present = fls.parse_frame_log(p)
        self.assertFalse(has_present)
        self.assertIsNone(frames[0].present)
        self.assertEqual(fls.read_frame_log(p), frames)


class StatsTest(unittest.TestCase):
    def test_percentile_is_linear_interpolation(self) -> None:
        v = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(fls.percentile(v, 0.5), 2.5)
        self.assertAlmostEqual(fls.percentile(v, 0.9), 3.7)
        self.assertEqual(fls.percentile([7.0], 0.99), 7.0)
        self.assertIsNone(fls.percentile([], 0.5))

    def test_summarize_empty(self) -> None:
        s = fls.summarize([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["mean"])

    def test_histogram_edges_are_left_closed(self) -> None:
        h = dict(fls.histogram([0, 4.99, 5, 24.9, 25, 59.99, 60, 80, 100], fls.DEFAULT_BINS))
        self.assertEqual(h, {"<5": 2, "5-25": 2, "25-45": 1, "45-60": 1, "60-80": 1, ">=80": 2})

    def test_vsync_histogram_nearest_multiple(self) -> None:
        v = 1000.0 / 60.0   # bin edges 8.33, 25, 41.67, 58.33, 75
        h = dict(fls.vsync_histogram([0.5, 8.3, 8.34, 16.7, 24.99, 25.0, 33.4, 50.0, 66.7, 74.9, 75.0, 120.0], v))
        self.assertEqual(h, {"0v": 2, "1v": 3, "2v": 2, "3v": 1, "4v": 2, ">=5v": 2})
        self.assertEqual(list(dict(fls.vsync_histogram([], 20.0, multiples=2))), ["0v", "1v", "2v", ">=3v"])

    def test_clusters(self) -> None:
        sizes, gaps = fls.clusters([1.0, 40.0, 2.0, 2.0, 50.0], 5.0)
        self.assertEqual(sizes, [2, 3, 1])
        self.assertEqual(gaps, [40.0, 50.0])
        self.assertEqual(fls.clusters([], 5.0), ([], []))


class FrameTest(unittest.TestCase):
    def frame(self, **kw) -> fls.Frame:
        base = dict(src=0, seq=0, dropped=0, arrival=None, decode_end=10.0, submit=12.0, npu_done=50.0,
                    gpu_post_done=None, complete=51.0)
        base.update(kw)
        return fls.Frame(**base)

    def test_host_finalize_uses_npu_done(self) -> None:
        m = fls.frame_metrics(self.frame())
        self.assertIsNone(m["e2e_arr"])
        self.assertIsNone(m["gpu_post"])
        self.assertEqual((m["e2e"], m["wait"], m["npu"], m["finalize"]), (41.0, 2.0, 38.0, 1.0))

    def test_gpu_path(self) -> None:
        m = fls.frame_metrics(self.frame(arrival=8.0, gpu_post_done=50.5))
        self.assertEqual((m["e2e_arr"], m["arr_dec"], m["gpu_post"], m["finalize"]), (43.0, 2.0, 0.5, 0.5))

    def test_non_monotonic(self) -> None:
        self.assertFalse(fls.non_monotonic(self.frame()))
        self.assertTrue(fls.non_monotonic(self.frame(submit=9.0)))

    def test_present_metrics_only_on_request(self) -> None:
        f = self.frame(arrival=8.0, present=60.0)
        self.assertNotIn("a2p", fls.frame_metrics(f))   # 9-column files keep the old keys
        m = fls.frame_metrics(f, with_present=True)
        self.assertEqual((m["a2p"], m["d2p"], m["display"]), (52.0, 50.0, 9.0))
        m = fls.frame_metrics(self.frame(), with_present=True)   # asap, not presented
        self.assertEqual((m["a2p"], m["d2p"], m["display"]), (None, None, None))

    def test_present_before_complete_is_non_monotonic(self) -> None:
        self.assertFalse(fls.non_monotonic(self.frame(present=51.0)))
        self.assertTrue(fls.non_monotonic(self.frame(present=50.5)))

    def test_drop_runs(self) -> None:
        fr = [self.frame(src=i, dropped=d) for i, d in enumerate([0, 1, 1, 1, 0, 2, 1, 0])]
        self.assertEqual(fls.drop_runs(fr), {"drop_runs": 2, "longest_drop_run": 3, "longest_queue_drop_run": 3})
        fr = [self.frame(src=i, dropped=d) for i, d in enumerate([0, 1, 2, 1, 0])]
        self.assertEqual(fls.drop_runs(fr), {"drop_runs": 1, "longest_drop_run": 3, "longest_queue_drop_run": 1})

    def test_batch_groups_chain_on_gap_and_cap(self) -> None:
        fr = [self.frame(src=i, submit=s, npu_done=n) for i, (s, n) in
              enumerate([(5, 10.0), (4, 11.9), (20, 60.0), (21, 62.0), (30, 110.0)])]
        g = fls.batch_groups(fr, 3.0)
        self.assertEqual([[f.src for f in grp] for grp in g], [[1, 0], [2, 3], [4]])   # ordered by submit
        self.assertEqual([len(grp) for grp in fls.batch_groups(fr, 3.0, cap=1)], [1] * 5)
        self.assertEqual([len(grp) for grp in fls.batch_groups(fr, 1.5)], [1, 1, 1, 1, 1])


class AnalyzeTest(Tmp):
    def analyze(self, rows: list[str], **kw) -> dict:
        fd = fls.load(self.write("p.csv", rows), "p.csv", 3.0, None)
        return fls.analyze([fd], "p.csv", **kw)

    def test_counts_and_drop_pattern(self) -> None:
        s = self.analyze(PACED_B2)
        c = s["counts"]
        self.assertTrue(s["paced"])
        self.assertEqual((c["rows"], c["processed"], c["dropped"], c["abandoned"]), (8, 5, 2, 1))
        self.assertEqual((c["drop_runs"], c["longest_drop_run"], c["longest_queue_drop_run"]), (2, 2, 1))
        self.assertEqual(c["non_monotonic_rows"], 0)

    def test_latency_and_stages(self) -> None:
        s = self.analyze(PACED_B2)
        e2e_arr = [103.0, 104.5 - 33.333, 203.5 - 100.0, 205.0 - 133.333, 302.0 - 166.667]
        self.assertEqual(s["latency"]["e2e_arr"]["n"], 5)
        self.assertAlmostEqual(s["latency"]["e2e_arr"]["mean"], sum(e2e_arr) / 5, places=6)
        self.assertAlmostEqual(s["latency"]["e2e_arr"]["max"], max(e2e_arr), places=6)
        self.assertAlmostEqual(s["stages"]["wait"]["max"], 6.0)          # src 3: 110 - 104
        self.assertAlmostEqual(s["stages"]["finalize"]["max"], 0.5)      # src 1: 104.5 - 104

    def test_cadence(self) -> None:
        s = self.analyze(PACED_B2)
        # complete: 103, 104.5, 203.5, 205, 302 -> diffs 1.5, 99, 1.5, 97
        cd = s["cadence"]
        self.assertEqual(cd["n"], 4)
        self.assertEqual(cd["histogram"], {"<5": 2, "5-25": 0, "25-45": 0, "45-60": 0, "60-80": 0, ">=80": 2})
        self.assertEqual(cd["clusters"], 3)
        self.assertEqual(cd["cluster_sizes"], {"1": 1, "2": 2})

    def test_batches(self) -> None:
        b = self.analyze(PACED_B2)["batches"]
        self.assertEqual(b["size_distribution"], {"1": 1, "2": 2})
        self.assertEqual(set(b["position_latency"]), {"0", "1", "single"})
        # first frames (src 0, 3) wait for their partner: e2e_arr 103, 103.5; second (src 1, 4): 71.167, 71.667
        self.assertAlmostEqual(b["position_latency"]["0"]["e2e_arr"]["mean"], 103.25, places=6)
        self.assertAlmostEqual(b["position_latency"]["1"]["e2e_arr"]["mean"], (71.167 + 71.667) / 2, places=6)
        self.assertAlmostEqual(b["partner_wait"]["mean"], (33.0 + 30.0) / 2)   # 38-5, 140-110
        self.assertAlmostEqual(b["npu_done_spread"]["max"], 2.0)

    def test_tail_flags(self) -> None:
        t = self.analyze(PACED_B2, top=3, startup=1, ending=3)["tail"]
        self.assertEqual(t["rank_by"], "e2e_arr")
        self.assertEqual([r["src"] for r in t["frames"]], [5, 3, 0])              # 135.333, 103.5, 103.0
        self.assertEqual([r["near_end"] for r in t["frames"]], [True, False, False])   # last 3 rows: src 5, 6, 7
        self.assertEqual([r["startup"] for r in t["frames"]], [False, False, True])
        self.assertEqual((t["startup_in_top"], t["near_end_in_top"]), (1, 1))
        self.assertEqual(t["frames"][1]["batch_pos"], 0)
        self.assertEqual(t["frames"][0]["batch_size"], 1)

    def test_asap_ranks_by_e2e(self) -> None:
        rows = [row(0, 0, 0, None, 5.0, 5.0, 50.0, None, 50.0), row(1, 1, 0, None, 9.0, 9.0, 52.0, None, 52.0)]
        s = self.analyze(rows, top=1)
        self.assertFalse(s["paced"])
        self.assertEqual(s["latency"]["e2e_arr"]["n"], 0)
        self.assertEqual(s["tail"]["rank_by"], "e2e")
        self.assertEqual(s["tail"]["frames"][0]["src"], 0)

    def test_no_present_keys_without_column(self) -> None:
        s = self.analyze(PACED_B2)
        self.assertNotIn("present_cadence", s)
        self.assertNotIn("presented", s["counts"])
        self.assertEqual(list(s["latency"]), list(fls.LATENCIES))
        self.assertEqual(list(s["stages"]), list(fls.STAGES))
        self.assertEqual(list(s["batches"]["position_latency"]["0"]), list(fls.LATENCIES))
        self.assertNotIn("present cadence", fls.format_report(s))


class PresentTest(Tmp):
    def analyze(self, files: dict[str, list[str]], **kw) -> dict:
        fds = [fls.load(self.write(name, rows, header=HEADER10 if rows is PACED_B2_DISP else HEADER), name, 3.0, None)
               for name, rows in files.items()]
        return fls.analyze(fds, "x", **kw)

    def test_counts(self) -> None:
        c = self.analyze({"d.csv": PACED_B2_DISP})["counts"]
        self.assertEqual((c["presented"], c["processed_not_presented"], c["present_on_unprocessed"]), (4, 1, 0))
        self.assertEqual(c["non_monotonic_rows"], 0)

    def test_latency_and_display_stage(self) -> None:
        s = self.analyze({"d.csv": PACED_B2_DISP})
        self.assertEqual(list(s["latency"]), ["e2e_arr", "e2e", "a2p", "d2p"])
        a2p = [110.0, 126.667 - 33.333, 210.0 - 100.0, 310.0 - 166.667]
        d2p = [105.0, 126.667 - 38.0, 106.0, 140.0]
        self.assertEqual(s["latency"]["a2p"]["n"], 4)
        self.assertAlmostEqual(s["latency"]["a2p"]["mean"], sum(a2p) / 4, places=6)
        self.assertAlmostEqual(s["latency"]["d2p"]["max"], max(d2p), places=6)
        self.assertEqual(s["latency"]["e2e"]["n"], 5)   # not-presented frames still count for e2e
        self.assertEqual(s["stages"]["display"]["n"], 4)
        self.assertAlmostEqual(s["stages"]["display"]["max"], 22.167, places=6)
        self.assertAlmostEqual(s["stages"]["display"]["median"], 7.5, places=6)

    def test_present_cadence(self) -> None:
        pc = self.analyze({"d.csv": PACED_B2_DISP})["present_cadence"]
        self.assertEqual(pc["n"], 3)
        self.assertEqual(pc["histogram"], {"0v": 0, "1v": 1, "2v": 0, "3v": 0, "4v": 0, ">=5v": 2})
        self.assertAlmostEqual(pc["quantiles"]["q50"], 83.333, places=6)
        self.assertAlmostEqual(pc["max"], 100.0, places=6)
        pc = self.analyze({"d.csv": PACED_B2_DISP}, vsync_ms=20.0)["present_cadence"]   # 50 Hz edges 10..90
        self.assertEqual(pc["vsync_ms"], 20.0)
        self.assertEqual(pc["histogram"], {"0v": 0, "1v": 1, "2v": 0, "3v": 0, "4v": 1, ">=5v": 1})

    def test_position_breakdown(self) -> None:
        pl = self.analyze({"d.csv": PACED_B2_DISP})["batches"]["position_latency"]
        self.assertEqual(list(pl["0"]), ["e2e_arr", "e2e", "a2p", "d2p"])
        self.assertEqual(pl["0"]["a2p"]["n"], 2)                               # src 0, 3
        self.assertAlmostEqual(pl["0"]["a2p"]["mean"], 110.0, places=6)
        self.assertEqual(pl["1"]["a2p"]["n"], 1)                               # src 1 (src 4 not presented)
        self.assertAlmostEqual(pl["1"]["d2p"]["mean"], 126.667 - 38.0, places=6)
        self.assertAlmostEqual(pl["single"]["a2p"]["mean"], 310.0 - 166.667, places=6)

    def test_tail_has_display(self) -> None:
        t = self.analyze({"d.csv": PACED_B2_DISP}, top=2)["tail"]
        self.assertEqual(t["rank_by"], "e2e_arr")                              # ranking unchanged
        self.assertEqual([r["src"] for r in t["frames"]], [5, 3])
        self.assertAlmostEqual(t["frames"][0]["display"], 8.0, places=6)

    def test_pooled_with_a_nine_column_file(self) -> None:
        s = self.analyze({"d.csv": PACED_B2_DISP, "p.csv": PACED_B2})
        c = s["counts"]
        # not-presented counts only rows of files that have the column
        self.assertEqual((c["processed"], c["presented"], c["processed_not_presented"]), (10, 4, 1))
        self.assertEqual(s["latency"]["a2p"]["n"], 4)
        self.assertEqual(s["present_cadence"]["n"], 3)

    def test_commit_columns_are_ignored(self) -> None:
        # yvo_pipeline appends commit_ms,present_seq after present_ms with --display on (12 columns, empty on the
        # Vulkan composition path). The stats must not change: same report as the 10-column file.
        rows12 = [r + (",," if i % 3 == 2 else f",{100.0 + i:.3f},{1000 + 2 * i}") for i, r in enumerate(PACED_B2_DISP)]
        f10 = fls.load(self.write("d10.csv", PACED_B2_DISP, header=HEADER10), "d.csv", 3.0, None)
        f12 = fls.load(self.write("d12.csv", rows12, header=HEADER10 + ",commit_ms,present_seq"), "d.csv", 3.0, None)
        self.assertTrue(f12.has_present)
        self.assertEqual(fls.format_report(fls.analyze([f12], "d.csv")), fls.format_report(fls.analyze([f10], "d.csv")))

    def test_present_on_dropped_row_warns(self) -> None:
        rows = list(PACED_B2_DISP)
        rows[2] = with_present(row(2, None, 1, 66.667, 70.0), 150.0)
        fd = fls.load(self.write("w.csv", rows, header=HEADER10), "w.csv", 3.0, None)
        s = fls.analyze([fd], "w.csv")
        self.assertEqual(s["counts"]["present_on_unprocessed"], 1)
        self.assertEqual(s["present_cadence"]["n"], 3)                         # processed frames only
        self.assertIn("warning: dropped/abandoned rows with present_ms 1", fls.format_report(s))


class CliTest(Tmp):
    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = fls.main(list(args))
        return rc, out.getvalue()

    def test_two_files_pooled_csv_json(self) -> None:
        a = self.write("a.csv", PACED_B2)
        b = self.write("b.csv", PACED_B2[:2])
        oc, oj = self.dir / "s.csv", self.dir / "s.json"
        rc, text = self.run_main(str(a), str(b), "--csv", str(oc), "--json", str(oj))
        self.assertEqual(rc, 0)
        self.assertIn("=== pooled", text)
        with open(oc, encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual([r["label"] for r in rows], ["a.csv", "b.csv", "pooled"])
        self.assertEqual(rows[2]["rows"], "10")
        self.assertEqual(rows[2]["processed"], "7")
        summ = json.loads(oj.read_text(encoding="utf-8"))["summaries"]
        self.assertEqual(summ[2]["files"], ["a.csv", "b.csv"])

    def test_present_csv_json_text(self) -> None:
        d = self.write("d.csv", PACED_B2_DISP, header=HEADER10)
        a = self.write("a.csv", PACED_B2)
        oc, oj = self.dir / "s.csv", self.dir / "s.json"
        rc, text = self.run_main(str(d), str(a), "--csv", str(oc), "--json", str(oj))
        self.assertEqual(rc, 0)
        self.assertEqual(text.count("present cadence"), 2)                     # d.csv and pooled, not a.csv
        self.assertIn("display: presented 4  processed but not presented 1 (20.0%)", text)
        self.assertIn("1v(~16.7): 1", text)
        self.assertIn("    present: a2p median 110.00", text)
        with open(oc, encoding="utf-8", newline="") as fh:
            rows = {r["label"]: r for r in csv.DictReader(fh)}
        self.assertEqual(rows["d.csv"]["processed_not_presented"], "1")
        self.assertEqual(rows["d.csv"]["present_cadence_>=5v"], "2")
        self.assertEqual(rows["d.csv"]["pos_0_a2p_median"], "110.0000")
        self.assertEqual(rows["d.csv"]["display_median"], "7.5000")
        self.assertEqual(rows["a.csv"]["present_cadence_n"], "")               # no column -> empty cells
        summ = json.loads(oj.read_text(encoding="utf-8"))["summaries"]
        self.assertIn("present_cadence", summ[0])
        self.assertNotIn("present_cadence", summ[1])
        self.assertIn("a2p", summ[2]["latency"])

    def test_bad_vsync(self) -> None:
        a = self.write("a.csv", PACED_B2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            fls.main([str(a), "--vsync-ms", "0"])
        self.assertEqual(cm.exception.code, 2)

    def test_refuses_to_overwrite_input(self) -> None:
        a = self.write("a.csv", PACED_B2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            fls.main([str(a), "--csv", str(a)])
        self.assertEqual(cm.exception.code, 2)

    def test_bad_input_exit_2(self) -> None:
        bad = self.write("bad.csv", ["x"], header="foo")
        rc, _ = self.run_main(str(bad))
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()
