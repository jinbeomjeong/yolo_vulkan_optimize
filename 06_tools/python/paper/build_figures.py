"""Regenerate every paper figure (10_paper/figures) from 07_results.

    /home/pi/miniconda3/envs/hailo-yolo/bin/python 06_tools/python/paper/build_figures.py
    ... build_figures.py --out-dir /tmp/figs          # test render somewhere else
    ... build_figures.py --only fig04 fig08           # a subset (stem prefix or full stem)

Each figure module is imported and its main(out_dir) is called; the figure's key numbers go to stdout.
No hardware is used. Figures drawn elsewhere (fig01 yEd diagram, ...) are not listed here.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_style as ps  # noqa: E402

FIGURES = (
    "fig02_npu_batch_fit",
    "fig03_npu_context_cost",
    "fig04_op_point_grid",
    "fig05_input_queue",
    "fig07_display_paths",
    "fig08_arrival_to_present",
    "fig09_thermal_soak",
)


def select(only: list[str] | None) -> list[str]:
    """Figure modules matching --only (exact stem or prefix such as 'fig04'). Unknown names are an error."""
    if not only:
        return list(FIGURES)
    chosen = []
    for name in only:
        hits = [f for f in FIGURES if f == name or f.startswith(name)]
        if not hits:
            raise SystemExit(f"error: no figure matches {name!r} (known: {', '.join(FIGURES)})")
        chosen += [h for h in hits if h not in chosen]
    return chosen


def build(out_dir: str | Path | None = None, only: list[str] | None = None) -> list[Path]:
    out = ps.resolve_out_dir(out_dir)
    written: list[Path] = []
    for name in select(only):
        mod = importlib.import_module(name)
        written += mod.main(out)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Regenerate the paper figures from 07_results.")
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default 10_paper/figures; relative paths are resolved from the repo root)")
    ap.add_argument("--only", nargs="+", metavar="FIG", help="figure stems or prefixes, e.g. fig04 fig08")
    a = ap.parse_args(argv)
    for p in build(a.out_dir, a.only):
        print(f"wrote: {ps.rel(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
