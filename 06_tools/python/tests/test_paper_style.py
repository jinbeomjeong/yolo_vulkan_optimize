"""Tests for 06_tools/python/paper/{paper_style,build_figures}.py (stdlib unittest, no pytest).

Run: /home/pi/miniconda3/envs/hailo-yolo/bin/python -m unittest discover -s 06_tools/python/tests
System python3 has no matplotlib: the whole module is skipped there. No 07_results file is read.
"""
from __future__ import annotations

import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path

PAPER = Path(__file__).resolve().parents[1] / "paper"
HAVE_MPL = importlib.util.find_spec("matplotlib") is not None and importlib.util.find_spec("numpy") is not None

if HAVE_MPL:
    sys.path.insert(0, str(PAPER))
    import build_figures as bf  # noqa: E402
    import paper_style as ps  # noqa: E402


@unittest.skipUnless(HAVE_MPL, "matplotlib/numpy not available (use the hailo-yolo conda env)")
class PaperStyleTest(unittest.TestCase):
    def test_median_min_max(self) -> None:
        self.assertEqual(ps.median_min_max([3.0, 1.0, 2.0]), (2.0, 1.0, 3.0))
        self.assertEqual(ps.median_min_max([1, 4]), (2.5, 1.0, 4.0))
        with self.assertRaises(ValueError):
            ps.median_min_max([])

    def test_linear_fit(self) -> None:
        a, b = ps.linear_fit([2, 4, 8], [24.8 + 16.2 * x for x in (2, 4, 8)])
        self.assertAlmostEqual(a, 24.8)
        self.assertAlmostEqual(b, 16.2)
        with self.assertRaises(ValueError):
            ps.linear_fit([1], [1])

    def test_fmt_half_up(self) -> None:
        self.assertEqual(f"{102.25:.1f}", "102.2")              # binary float rounding: the bug being avoided
        self.assertEqual(ps.fmt_half_up(102.25), "102.3")
        self.assertEqual(ps.fmt_half_up(81.85), "81.9")
        self.assertEqual(ps.fmt_half_up(24.775, 2), "24.78")
        self.assertEqual(ps.fmt_half_up(120.5, 0), "121")
        self.assertEqual(ps.fmt_half_up(3.0, 1), "3.0")

    def test_canonical_clip_ids(self) -> None:
        def svg(i1: str, i2: str) -> str:
            return (f'<g clip-path="url(#{i1})"/><g clip-path="url(#{i2})"/><g id="patch_1"/>'
                    f'<clipPath id="{i1}"><rect x="1" y="2"/></clipPath>'
                    f'<clipPath id="{i2}"><rect x="1" y="2"/></clipPath>')
        a = ps.canonical_clip_ids(svg("p0123456789", "pabcdefabcd"))
        b = ps.canonical_clip_ids(svg("pfedcba9876", "p1111111111"))
        self.assertEqual(a, b)                      # same printed geometry -> same ids, whatever the input ids
        self.assertIn('id="patch_1"', a)            # other ids untouched
        ids = sorted(set(re.findall(r'clipPath id="(c[0-9a-f]+)"', a)))
        self.assertEqual(len(ids), 2)               # identical bodies still get distinct ids
        self.assertEqual(ps.canonical_clip_ids("<svg/>"), "<svg/>")

    def test_resolve_out_dir(self) -> None:
        self.assertEqual(ps.resolve_out_dir(None), ps.REPO_ROOT / "10_paper" / "figures")
        self.assertEqual(ps.resolve_out_dir("x/y"), ps.REPO_ROOT / "x" / "y")
        self.assertEqual(ps.resolve_out_dir("/tmp/abs"), Path("/tmp/abs"))

    def test_new_figure_width(self) -> None:
        fig, _ = ps.new_figure("single", 5.0)
        self.assertAlmostEqual(fig.get_figwidth() * 2.54, 8.5)
        ps.plt.close(fig)
        with self.assertRaises(ValueError):
            ps.new_figure("triple")

    def test_save_is_byte_identical(self) -> None:
        def render(out: Path) -> list[bytes]:
            fig, ax = ps.new_figure("single", 4.0)
            ps.errorbar_median(ax, [1, 2], [[1.0, 2.0, 3.0], [2.0, 2.5, 4.0]], fmt="o", label="중앙값")
            ax.set_xlabel("배치 크기 B")
            ax.legend()
            return [p.read_bytes() for p in ps.save(fig, out, "fig99_test")]

        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            first, second = render(Path(d1)), render(Path(d2))
            self.assertEqual(first, second)
            self.assertNotIn(b"<dc:date>", first[0])
            self.assertIn(b"font-family", first[0])   # svg.fonttype none: text stays text


@unittest.skipUnless(HAVE_MPL, "matplotlib/numpy not available (use the hailo-yolo conda env)")
class BuildFiguresTest(unittest.TestCase):
    def test_every_listed_figure_has_a_script_with_main(self) -> None:
        for name in bf.FIGURES:
            self.assertTrue((PAPER / f"{name}.py").is_file(), name)
        self.assertEqual(sorted(p.stem for p in PAPER.glob("fig*.py")), sorted(bf.FIGURES))

    def test_select(self) -> None:
        self.assertEqual(bf.select(None), list(bf.FIGURES))
        self.assertEqual(bf.select(["fig04", "fig04_op_point_grid"]), ["fig04_op_point_grid"])
        with self.assertRaises(SystemExit):
            bf.select(["fig99"])


if __name__ == "__main__":
    unittest.main()
