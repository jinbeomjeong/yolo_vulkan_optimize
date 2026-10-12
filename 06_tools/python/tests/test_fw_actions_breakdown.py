"""Tests for 06_tools/python/fw_actions_breakdown.py (stdlib unittest, no pytest).

Run: python3 -m unittest discover -s 06_tools/python/tests

The synthetic JSON mirrors the measure-fw-actions structure of the 2026-10-08 probe (runs[0].contexts[] with
activation / preliminary / dynamic_N / batch_switching, 200 MHz cycle timestamps). Spans are built so that
  dynamic_0 = 2 ms (module_config_done_interrupt) + 1 ms * B (lcu_interrupt)
  dynamic_1 = 1 ms (module_config_done_interrupt) + 2 ms * B (lcu_interrupt)
with a 0.01 ms switch gap, so the fits must come out as a=2,b=1 / a=1,b=2 / total a=3,b=3.
One optional test reads the real probe files in 11_logs/ (git-ignored) and is skipped when they are absent.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fw_actions_breakdown as fab  # noqa: E402

CLK = 200            # MHz
CYC = CLK * 1000     # cycles per ms
REPO = Path(__file__).resolve().parents[3]


def act(t_ms: float, typ: str, **data) -> dict:
    return {"address": 0, "timestamp": round(t_ms * CYC), "type": typ, "data": data}


def ctx(name: str, actions: list[dict], batch_counter: int = 1) -> dict:
    return {"action_list_base_address": 0, "action_list_size": 0, "batch_counter": batch_counter,
            "idle_time": 0, "context_name": name, "actions": actions}


def make_json(B: int, *, swap_dynamic_order: bool = False) -> dict:
    t0 = 100.0
    d0 = [act(t0, "activate_cfg_channel"), act(t0 + 2.0, "module_config_done_interrupt"),
          act(t0 + 2.0 + 1.0 * B, "lcu_interrupt")]
    s1 = t0 + 2.0 + 1.0 * B + 0.01
    if swap_dynamic_order:
        s1 = t0 - 50.0          # dynamic_1 recorded before dynamic_0: not back to back
    d1 = [act(s1, "activate_cfg_channel"), act(s1 + 1.0, "module_config_done_interrupt"),
          act(s1 + 1.0 + 2.0 * B, "lcu_interrupt")]
    period = 3.0 + 3.0 * B + 0.5
    return {
        "version": "2.0", "creation_time": "synthetic", "clock_cycle_MHz": CLK,
        "hef": {"path": "x.hef", "file_hash": "0"},
        "runs": [{
            "batch_size": B, "mean_activation_time_ms": 0.1, "mean_deactivation_time_ms": 0.05,
            "network_group_id": 0, "fps": 1000.0 * B / period,
            "contexts": [
                ctx("activation", [act(0.001, "burst_credits_task_reset"), act(0.002, "open_boundary_input_channel")]),
                ctx("batch_switching", [act(0, "switch_lcu_batch"), act(0, "burst_credits_task_start")], 0),
                ctx("preliminary", [act(0.1, "activate_cfg_channel"), act(5.1, "module_config_done_interrupt")]),
                ctx("dynamic_1", d1, 10),     # out of order on purpose: the tool sorts by index
                ctx("dynamic_0", d0, 11),
            ],
        }],
    }


class Tmp(unittest.TestCase):
    def setUp(self) -> None:
        self._td = tempfile.TemporaryDirectory()
        self.dir = Path(self._td.name)

    def tearDown(self) -> None:
        self._td.cleanup()

    def write(self, name: str, obj) -> Path:
        p = self.dir / name
        p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj, encoding="utf-8")
        return p


class RunTest(Tmp):
    def result(self, B: int = 1, **kw) -> dict:
        (run,) = fab.load(self.write(f"b{B}.json", make_json(B, **kw)))
        return fab.analyze_run(run)

    def test_spans_and_types(self) -> None:
        r = self.result(4)
        by = {c["name"]: c for c in r["contexts"]}
        self.assertEqual(r["dynamic"], ["dynamic_0", "dynamic_1"])
        self.assertAlmostEqual(by["dynamic_0"]["span_ms"], 6.0)
        self.assertAlmostEqual(by["dynamic_1"]["span_ms"], 9.0)
        self.assertAlmostEqual(by["dynamic_0"]["type_ms"]["module_config_done_interrupt"], 2.0)
        self.assertAlmostEqual(by["dynamic_0"]["type_ms"]["lcu_interrupt"], 4.0)
        for c in r["contexts"]:
            if c["timed"]:
                self.assertAlmostEqual(sum(c["type_ms"].values()), c["span_ms"])
        self.assertEqual(by["dynamic_1"]["batch_counter"], 10)

    def test_batch_switching_not_timed(self) -> None:
        c = {c["name"]: c for c in self.result()["contexts"]}["batch_switching"]
        self.assertFalse(c["timed"])
        self.assertIsNone(c["span_ms"])

    def test_batch_span_and_switch_gap(self) -> None:
        r = self.result(2)
        self.assertTrue(r["contiguous"])
        self.assertAlmostEqual(r["switch_gaps_ms"][0], 0.01)
        self.assertAlmostEqual(r["dynamic_span_sum_ms"], 9.0)
        self.assertAlmostEqual(r["batch_span_ms"], 9.01)
        self.assertAlmostEqual(r["fps_period_ms"], 9.5)

    def test_not_back_to_back(self) -> None:
        r = self.result(2, swap_dynamic_order=True)
        self.assertFalse(r["contiguous"])
        self.assertIsNone(r["batch_span_ms"])

    def test_largest_gaps(self) -> None:
        c = {c["name"]: c for c in self.result(4)["contexts"]}["dynamic_1"]
        self.assertEqual(c["largest_gaps"][0]["before"], "lcu_interrupt")
        self.assertAlmostEqual(c["largest_gaps"][0]["ms"], 8.0)

    def test_counter_wrap_is_unwrapped(self) -> None:
        d = make_json(1)
        acts = d["runs"][0]["contexts"][4]["actions"]           # dynamic_0
        base = fab.WRAP - round(1.0 * CYC)                      # first action 1 ms before the wrap
        for i, a in enumerate(acts):
            a["timestamp"] = (base + i * round(1.5 * CYC)) % fab.WRAP
        (run,) = fab.load(self.write("w.json", d))
        c = {c.name: c for c in run.contexts}["dynamic_0"]
        self.assertEqual(c.wraps, 1)
        self.assertAlmostEqual(run.ms(c.actions[-1][0] - c.actions[0][0]), 3.0)

    def test_repeated_context_name(self) -> None:
        d = make_json(1)
        d["runs"][0]["contexts"].append(ctx("dynamic_0", [act(500.0, "activate_cfg_channel")]))
        (run,) = fab.load(self.write("r.json", d))
        self.assertIn("dynamic_0#2", [c.name for c in run.contexts])

    def test_bad_structure(self) -> None:
        with self.assertRaises(fab.FwActionsError):
            fab.load(self.write("x.json", {"runs": []}))
        with self.assertRaises(fab.FwActionsError):
            fab.load(self.write("y.json", "not json"))
        d = make_json(1)
        del d["runs"][0]["contexts"][0]["actions"][0]["timestamp"]
        with self.assertRaises(fab.FwActionsError):
            fab.load(self.write("z.json", d))


class FitTest(Tmp):
    def test_fit_affine(self) -> None:
        self.assertIsNone(fab.fit_affine([(1, 2.0), (1, 3.0)]))
        f = fab.fit_affine([(1, 3.0), (8, 10.0)])
        self.assertAlmostEqual(f["a"], 2.0)
        self.assertAlmostEqual(f["b"], 1.0)
        f = fab.fit_affine([(1, 1.0), (1, 3.0), (3, 6.0)])    # repeats: mean of B=1 is 2
        self.assertAlmostEqual(f["a"] + f["b"], 2.0)
        self.assertAlmostEqual(f["a"] + 3 * f["b"], 6.0)
        self.assertGreater(f["rms"], 0)

    def test_fit_all_recovers_model(self) -> None:
        results = [fab.analyze_run(r) for B in (1, 2, 8) for r in fab.load(self.write(f"b{B}.json", make_json(B)))]
        fits = fab.fit_all(results, 0.5)
        self.assertEqual(list(fits["contexts"]), ["dynamic_0", "dynamic_1"])
        for name, (a, b) in {"dynamic_0": (2.0, 1.0), "dynamic_1": (1.0, 2.0)}.items():
            self.assertAlmostEqual(fits["contexts"][name]["a"], a, places=6)
            self.assertAlmostEqual(fits["contexts"][name]["b"], b, places=6)
        self.assertAlmostEqual(fits["totals"]["dynamic_span_sum"]["a"], 3.0, places=6)
        self.assertAlmostEqual(fits["totals"]["batch_span"]["a"], 3.01, places=6)
        self.assertAlmostEqual(fits["totals"]["fps_period"]["b"], 3.0, places=6)
        cfg = fits["types"]["dynamic_0"]["module_config_done_interrupt"]
        lcu = fits["types"]["dynamic_0"]["lcu_interrupt"]
        self.assertAlmostEqual(cfg["a"], 2.0, places=6)
        self.assertAlmostEqual(cfg["b"], 0.0, places=6)
        self.assertAlmostEqual(lcu["a"], 0.0, places=6)
        self.assertAlmostEqual(lcu["b"], 1.0, places=6)

    def test_single_batch_size_has_no_fit(self) -> None:
        results = [fab.analyze_run(r) for r in fab.load(self.write("b1.json", make_json(1)))]
        fits = fab.fit_all(results, 0.5)
        self.assertIsNone(fits["contexts"]["dynamic_0"])
        self.assertIn("needs >= 2 distinct B", fab.format_fits(fits, results, fab.REF_MODEL))


class CliTest(Tmp):
    def run_main(self, *args: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = fab.main(list(args))
        return rc, out.getvalue()

    def test_text_and_json(self) -> None:
        a, b = self.write("b1.json", make_json(1)), self.write("b8.json", make_json(8))
        oj = self.dir / "out.json"
        rc, text = self.run_main(str(a), str(b), "--json", str(oj), "--ref", "3,3")
        self.assertEqual(rc, 0)
        self.assertIn("reference T(B) = 3 + 3*B", text)
        self.assertIn("inferred", text)
        d = json.loads(oj.read_text(encoding="utf-8"))
        self.assertEqual([r["batch_size"] for r in d["runs"]], [1, 8])
        self.assertAlmostEqual(d["fits"]["totals"]["dynamic_span_sum"]["b"], 3.0, places=6)

    def test_refuses_to_overwrite_input(self) -> None:
        a = self.write("b1.json", make_json(1))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
            fab.main([str(a), "--json", str(a)])
        self.assertEqual(cm.exception.code, 2)

    def test_bad_ref(self) -> None:
        a = self.write("b1.json", make_json(1))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            fab.main([str(a), "--ref", "1"])

    def test_bad_input_exit_2(self) -> None:
        rc, _ = self.run_main(str(self.write("bad.json", "{")))
        self.assertEqual(rc, 2)


B1 = REPO / "11_logs" / "2026-10-08_fwactions_probe_b1.json"
B8 = REPO / "11_logs" / "2026-10-08_fwactions_probe_b8.json"


@unittest.skipUnless(B1.is_file() and B8.is_file(), "2026-10-08 probe JSONs not present (11_logs is git-ignored)")
class RealProbeTest(unittest.TestCase):
    def test_matches_hand_analysis(self) -> None:
        r1, r8 = (fab.analyze_run(fab.load(p)[0]) for p in (B1, B8))
        self.assertAlmostEqual(r1["dynamic_span_sum_ms"], 40.92, delta=0.01)
        self.assertAlmostEqual(r8["dynamic_span_sum_ms"], 154.54, delta=0.01)
        spans8 = [c["span_ms"] for c in r8["contexts"] if c["name"] in r8["dynamic"]]
        for got, want in zip(spans8, (50.10, 35.60, 49.65, 19.19)):
            self.assertAlmostEqual(got, want, delta=0.01)


if __name__ == "__main__":
    unittest.main()
