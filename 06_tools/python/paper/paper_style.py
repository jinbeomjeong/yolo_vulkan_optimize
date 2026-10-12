"""Shared matplotlib style and helpers for the paper figures (10_paper/figures).

Rules (.claude/skills/paper-figures/SKILL.md §3):
  - Agg backend, conda env hailo-yolo (system python3 has no matplotlib).
  - Font NanumSquare, UnDotum when it is missing; svg.fonttype 'none', pdf.fonttype 42.
  - Single column 8.5 cm, double column 17.5 cm, text 8-9 pt.
  - Processing-unit colors follow 09_diagrams/README.md; other series use a colorblind-safe palette (Okabe-Ito).
  - Repeated measurements are drawn as median with a min-max range.
  - Fixed svg.hashsalt and empty metadata dates, so the same input gives byte-identical SVG output.

Every figure script exposes main(out_dir) -> list[Path] and a CLI with --out-dir (see figure_cli()).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections.abc import Callable, Iterable, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "07_results"
DEFAULT_OUT_DIR = REPO_ROOT / "10_paper" / "figures"

CM = 1.0 / 2.54                      # inches per cm
SINGLE_COL_CM = 8.5
DOUBLE_COL_CM = 17.5
FONT_SIZE = 8.0                      # body text, ticks, legends
TITLE_SIZE = 9.0                     # panel titles, axis labels
PNG_DPI = 300
SVG_HASHSALT = "yvo-paper-figures"

FONT_PREFERENCE = ("NanumSquare", "UnDotum")
FONT_LAST_RESORT = "DejaVu Sans"     # glyphs neither Korean font has (e.g. some math symbols)

# Processing-unit colors (09_diagrams/README.md color convention)
DEVICE_COLORS = {
    "gpu": "#99CC00",       # V3D GPU, Vulkan compute
    "npu": "#FFCC00",       # Hailo-8 NPU
    "cpu": "#E8EEF7",       # CPU main thread
    "decoder": "#B39DDB",   # HW HEVC decoder
    "display": "#4DB6E2",   # display: swapchain present, HVS plane scanout
}
EDGE_COLOR = "#333333"
GRID_COLOR = "#D9D9D9"
REFERENCE_COLOR = "#7F7F7F"          # reference lines (30 fps, fits)

# Okabe-Ito colorblind-safe palette for series that are not processing units
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000")


def available_fonts() -> list[str]:
    """Preferred Korean fonts that matplotlib can find, in preference order, then the last-resort font."""
    names = {f.name for f in font_manager.fontManager.ttflist}
    chosen = [n for n in FONT_PREFERENCE if n in names]
    if not chosen:
        print(f"warning: none of {', '.join(FONT_PREFERENCE)} found; Korean text will not render",
              file=sys.stderr)
    return chosen + [FONT_LAST_RESORT]


def apply_style() -> None:
    """Set the shared rcParams. Called by every figure before drawing."""
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": available_fonts(),
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 0.5,
        "legend.frameon": False,
        "axes.unicode_minus": False,     # the Korean fonts lack U+2212
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": EDGE_COLOR,
        "axes.linewidth": 0.6,
        "axes.grid": True,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.0,
        "errorbar.capsize": 2.0,
        "svg.fonttype": "none",
        "svg.hashsalt": SVG_HASHSALT,
        "pdf.fonttype": 42,
        "savefig.dpi": PNG_DPI,
        "figure.dpi": 100,
    })


def new_figure(width: str = "single", height_cm: float = 6.0, **subplots_kw) -> tuple[Figure, object]:
    """Figure of single (8.5 cm) or double (17.5 cm) column width. Returns (fig, axes) like plt.subplots."""
    match width:
        case "single":
            w = SINGLE_COL_CM
        case "double":
            w = DOUBLE_COL_CM
        case _:
            raise ValueError(f"width must be 'single' or 'double', got {width!r}")
    apply_style()
    subplots_kw.setdefault("layout", "constrained")
    return plt.subplots(figsize=(w * CM, height_cm * CM), **subplots_kw)


def save(fig: Figure, out_dir: str | Path, stem: str) -> list[Path]:
    """Write <stem>.svg and <stem>.png (300 dpi) into out_dir. Metadata dates are left out for reproducibility."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    svg = out / f"{stem}.svg"
    png = out / f"{stem}.png"
    fig.savefig(svg, format="svg", metadata={"Date": None})
    svg.write_text(canonical_clip_ids(svg.read_text(encoding="utf-8")), encoding="utf-8")
    fig.savefig(png, format="png", dpi=PNG_DPI, metadata={"Software": None})
    plt.close(fig)
    return [svg, png]


_CLIP_RE = re.compile(r'<clipPath id="(p[0-9a-f]+)">(.*?)</clipPath>', re.S)


def canonical_clip_ids(svg_text: str) -> str:
    """Re-derive matplotlib's SVG clip-path ids from the printed <clipPath> body.

    matplotlib hashes the full-precision float clip rectangle (plus svg.hashsalt). Constrained layout can differ in
    the last bits between processes (seen in fig09 when all figures are built in one process), which changes the id
    although the printed rectangle is identical. Ids made from the printed text keep the SVG byte-identical.
    """
    mapping: dict[str, str] = {}
    seen: dict[str, int] = {}
    for m in _CLIP_RE.finditer(svg_text):
        body = m.group(2)
        n = seen.get(body, 0)
        seen[body] = n + 1
        mapping[m.group(1)] = "c" + hashlib.sha256(f"{n}:{body}".encode()).hexdigest()[:10]
    if not mapping:
        return svg_text
    return re.sub(r'(id="|url\(#)(p[0-9a-f]+)\b',
                  lambda m: m.group(1) + mapping.get(m.group(2), m.group(2)), svg_text)


# ─── Statistics ─────────────────────────────────────────────────────────────────
def median_min_max(values: Iterable[float]) -> tuple[float, float, float]:
    """(median, min, max) of repeated measurements (the measure skill's aggregation)."""
    v = [float(x) for x in values]
    if not v:
        raise ValueError("median_min_max: no values")
    return statistics.median(v), min(v), max(v)


def errorbar_median(ax: Axes, x: Sequence[float], samples: Sequence[Sequence[float]], **kw):
    """Plot the median of each sample set at x with a min-max error bar. Returns the ErrorbarContainer."""
    stats = [median_min_max(s) for s in samples]
    med = [m for m, _, _ in stats]
    lower = [m - lo for m, lo, _ in stats]
    upper = [hi - m for m, _, hi in stats]
    kw.setdefault("capsize", plt.rcParams["errorbar.capsize"])
    kw.setdefault("elinewidth", 0.8)
    return ax.errorbar(list(x), med, yerr=[lower, upper], **kw)


def fmt_half_up(x: float, nd: int = 1) -> str:
    """Round half up on the decimal text of x (102.25 -> '102.3'), the convention of the 07_results summaries.
    Plain f'{x:.1f}' rounds the binary float (102.25 -> '102.2'), so figure labels use this instead."""
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def linear_fit(x: Sequence[float], y: Sequence[float]) -> tuple[float, float]:
    """Least-squares y = a + b*x. Returns (a, b)."""
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("linear_fit: need at least two (x, y) pairs")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    sxx = sum((xi - mx) ** 2 for xi in x)
    sxy = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    b = sxy / sxx
    return my - b * mx, b


# ─── Inputs ─────────────────────────────────────────────────────────────────────
def read_csv(path: str | Path) -> list[dict[str, str]]:
    """All rows of a CSV as dicts (07_results files are UTF-8)."""
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path: str | Path) -> object:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def rel(path: str | Path) -> str:
    """Repo-relative path for audit output (absolute when outside the repo)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(p)


def report(fig_stem: str, sources: Iterable[str | Path], lines: Iterable[str]) -> None:
    """Print the plotted key numbers so they can be audited against 07_results summaries."""
    print(f"== {fig_stem}")
    for s in sources:
        print(f"  source: {rel(s)}")
    for ln in lines:
        print(f"  {ln}")


# ─── CLI ────────────────────────────────────────────────────────────────────────
def resolve_out_dir(out_dir: str | Path | None) -> Path:
    """None -> 10_paper/figures; a relative path is resolved from the repo root (not the CWD)."""
    if out_dir is None:
        return DEFAULT_OUT_DIR
    p = Path(out_dir)
    return p if p.is_absolute() else REPO_ROOT / p


def figure_cli(main: Callable[[Path | None], list[Path]], description: str,
               argv: list[str] | None = None) -> int:
    """Shared CLI of the figure scripts: --out-dir (default 10_paper/figures, relative to the repo root)."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--out-dir", default=None,
                    help="output folder (default 10_paper/figures; relative paths are resolved from the repo root)")
    a = ap.parse_args(argv)
    for p in main(resolve_out_dir(a.out_dir)):
        print(f"  wrote: {p}")
    return 0
