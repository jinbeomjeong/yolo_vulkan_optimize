"""Figure 14: why the CPU reading GPU memory is expensive. Time per 1,228,800 B access, by access pattern and memory type.

The cost is in small CPU loads from non-cached memory; CPU writes to that memory and GPU reads of it are cheap.
Linear axis, so the size of the difference stays visible. Bars are medians of per-run medians with a min-max range over 3 runs.
Sources (under 07_results):
  CPU reads   gpu_postprocess/runs/stride_read/evidence/stride_run{1,2,3}.txt: per-run median_ms of the fixed-format table.
              Rows v3d seq_byte, v3d seq_memcpy, v3d and malloc stride3_byte; heap stride3_byte goes into a label because
              summary.md §4 calls malloc an optimistic reference.
  CPU writes  memory_asymmetry/raw/cpuwrite_run{1,2,3}/stdout.log: v3d memcpy, v3d seq_byte; checked against evidence/medians.txt §4.
  GPU read    memory_asymmetry/raw/bw_r256x3_run{1,2,3}/stdout.log: v3d read_v4 at 1,228,800 B, raw GB/s (with the per-dispatch
              fixed cost), checked against evidence/medians.txt §1. Converted to ms: it was measured as a bandwidth.
              The bandwidth depends strongly on the dispatch shape (memory_asymmetry/summary.md §4); the caption says so.
The dates, governors and V3D clocks of the two sessions are read from each env_bench.txt.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

import paper_style as ps

STEM = "fig14_memory_read_cost"
STRIDE_DIR = ps.RESULTS / "gpu_postprocess" / "runs" / "stride_read"
ASYM_DIR = ps.RESULTS / "memory_asymmetry"
MEDIANS = ASYM_DIR / "evidence" / "medians.txt"
RUNS = (1, 2, 3)
BUFFER_BYTES = 1228800               # checked against both sessions in load()

GPU_MEM_COLOR = ps.PALETTE[3]        # memory types are not processing units: Okabe-Ito, not the device colors
CACHED_MEM_COLOR = ps.PALETTE[0]
GPU_MEM_LABEL = "GPU 메모리(V3D, 비캐시 매핑)"
CACHED_MEM_LABEL = "일반 캐시 메모리(malloc, 낙관적 기준)"
CONVERTED_HATCH = "////"

BAR_HEIGHT = 0.56
PAIR_HEIGHT = 0.34                   # the two bars of the transpose row
X_HEADROOM = 1.3                     # room for the labels and the date column right of the longest bar
X_TICK_STEP = 2.0
LABEL_GAP = 0.008                    # share of x_max between a whisker end and its value label
AXES_LEFT, AXES_RIGHT, AXES_BOTTOM, AXES_TOP = 0.2, 0.985, 0.25, 0.93
HATCH_WIDTH = 0.45
NOTE_X = 0.008
NOTE_Y = 0.012                       # figure coordinates
MEDIANS_TOLERANCE = 0.0006           # 3-decimal text


@dataclass(frozen=True)
class Value:
    median: float
    low: float
    high: float


def _check(what: str, got: float, expected: float, tolerance: float) -> None:
    if abs(got - expected) > tolerance:
        raise ValueError(f"{what}: {got} does not match {expected} (tolerance {tolerance})")


def _stats(samples: list[float]) -> Value:
    return Value(*ps.median_min_max(samples))


def _search(path: Path, pattern: str, text: str | None = None) -> re.Match[str]:
    m = re.search(pattern, path.read_text(encoding="utf-8") if text is None else text, re.M)
    if m is None:
        raise ValueError(f"{ps.rel(path)}: no line matches {pattern!r}")
    return m


def load_stride() -> dict[tuple[str, str], Value]:
    """Median of the per-run median_ms per (alloc, pattern). Raises ValueError on a failed run or another buffer size."""
    samples: dict[tuple[str, str], list[float]] = {}
    for run in RUNS:
        path = STRIDE_DIR / "evidence" / f"stride_run{run}.txt"
        text = path.read_text(encoding="utf-8")
        _search(path, r"^\[RESULT\] stride PASS$", text)
        _search(path, r"^\[validation\] errors=0 ", text)
        _check(f"{ps.rel(path)} buffer", int(_search(path, r"= (\d+) B ", text).group(1)), BUFFER_BYTES, 0)
        rows = re.findall(r"^(v3d|malloc|heap)\s+(seq_memcpy|seq_byte|stride3_byte)\s+([\d.]+)\s", text, re.M)
        if len(rows) != 9:
            raise ValueError(f"{ps.rel(path)}: expected 9 table rows, found {len(rows)}")
        for alloc, pattern, ms in rows:
            samples.setdefault((alloc, pattern), []).append(float(ms))
    return {key: _stats(v) for key, v in samples.items()}


def _medians_section(heading: str) -> str:
    """One '## ...' section of memasym_aggregate.py's medians.txt."""
    sections = re.split(r"^## ", MEDIANS.read_text(encoding="utf-8"), flags=re.M)
    hits = [s for s in sections if s.startswith(heading)]
    if len(hits) != 1:
        raise ValueError(f"{ps.rel(MEDIANS)}: section {heading!r} found {len(hits)} times")
    return hits[0]


def _check_passed_runs() -> None:
    text = MEDIANS.read_text(encoding="utf-8")
    line = _search(MEDIANS, r"^results: (.*)$", text).group(1)
    if "FAIL" in line or line.count("PASS") != 15:
        raise ValueError(f"{ps.rel(MEDIANS)}: not all 15 runs passed: {line}")


def load_cpu_writes() -> dict[str, Value]:
    """CPU write times to V3D memory per pattern, from the per-run logs, checked against medians.txt §4."""
    samples: dict[str, list[float]] = {"memcpy": [], "seq_byte": []}
    for run in RUNS:
        path = ASYM_DIR / "raw" / f"cpuwrite_run{run}" / "stdout.log"
        text = path.read_text(encoding="utf-8")
        _check(f"{ps.rel(path)} buffer", int(_search(path, r"CPU write of (\d+) B", text).group(1)), BUFFER_BYTES, 0)
        for pattern, values in samples.items():
            values.append(float(_search(path, rf"^v3d\s+{pattern}\s+([\d.]+)\s", text).group(1)))
    out = {key: _stats(v) for key, v in samples.items()}
    section = _medians_section("4. CPU write")
    for pattern, v in out.items():
        m = _search(MEDIANS, rf"^v3d\s+{pattern}\s+([\d.]+) \[([\d.]+)\.\.([\d.]+)\]", section)
        for got, text_value in zip((v.median, v.low, v.high), m.groups()):
            _check(f"medians.txt v3d {pattern}", got, float(text_value), MEDIANS_TOLERANCE)
    return out


def _gbps_to_ms(gbps: float) -> float:
    """Time of one buffer at this bandwidth: GB/s is bytes per ns, so bytes / (GB/s) is ns."""
    return BUFFER_BYTES / gbps / 1e6


def _two(x: float) -> str:
    return ps.fmt_half_up(x, 2)


def load_gpu_read() -> tuple[Value, Value, float]:
    """V3D read_v4 of the 1.2 MB buffer: (time in ms from raw GB/s, the raw GB/s, the net GB/s median for the caption)."""
    raw: list[float] = []
    for run in RUNS:
        path = ASYM_DIR / "raw" / f"bw_r256x3_run{run}" / "stdout.log"
        raw.append(float(_search(path, rf"^{BUFFER_BYTES}\s+read_v4\s+v3d\s+([\d.]+)\s").group(1)))
    gbps = _stats(raw)
    m = _search(MEDIANS, rf"^bw_r256x3\s+{BUFFER_BYTES}\s+read_v4\s+v3d\s+([\d.]+) \[[\d.]+\.\.[\d.]+\]\s+([\d.]+)",
                _medians_section("1. GPU bandwidth"))
    _check("medians.txt v3d read_v4", gbps.median, float(m.group(1)), MEDIANS_TOLERANCE)
    return Value(_gbps_to_ms(gbps.median), _gbps_to_ms(gbps.high), _gbps_to_ms(gbps.low)), gbps, float(m.group(2))


def load_sessions() -> dict[str, dict[str, str]]:
    """Date, CPU governor and V3D clock of the two sessions, from their env_bench.txt."""
    out: dict[str, dict[str, str]] = {}
    for key, path, governor_re in (("stride", STRIDE_DIR / "env_bench.txt", r"^governor cpu\d: (\w+)"),
                                   ("asym", ASYM_DIR / "env_bench.txt", r"^governor\(cpu0\.\.3\): (\w+)")):
        text = path.read_text(encoding="utf-8")
        out[key] = {"date": _search(path, r"^date: (\d{4}-\d{2}-\d{2})", text).group(1),
                    "governor": _search(path, governor_re, text).group(1),
                    "v3d": _search(path, r"v3d_freq=(\d+)", text).group(1)}
    return out


def _bar(ax, y: float, height: float, v: Value, color: str, x_max: float, text: str, hatch: str | None = None) -> None:
    ax.barh(y, v.median, height, color=color, edgecolor=ps.EDGE_COLOR, lw=0.4, hatch=hatch,
            xerr=[[v.median - v.low], [v.high - v.median]], error_kw={"elinewidth": 0.7, "capsize": 1.5, "ecolor": ps.EDGE_COLOR})
    ax.text(v.high + x_max * LABEL_GAP, y, text, ha="left", va="center")


def main(out_dir: str | Path | None = None) -> list[Path]:
    _check_passed_runs()
    stride, writes, sessions = load_stride(), load_cpu_writes(), load_sessions()
    gpu_ms, gpu_gbps, gpu_net = load_gpu_read()
    read_date, asym_date = sessions["stride"]["date"], sessions["asym"]["date"]

    fig, ax = ps.new_figure("double", 8.2, layout="none")
    fig.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, bottom=AXES_BOTTOM, top=AXES_TOP)
    x_max = stride[("v3d", "seq_byte")].high * X_HEADROOM
    plt.rcParams["hatch.linewidth"] = HATCH_WIDTH

    gpu_byte, gpu_tr, cached_tr = stride[("v3d", "seq_byte")], stride[("v3d", "stride3_byte")], stride[("malloc", "stride3_byte")]
    heap_tr, gpu_copy = stride[("heap", "stride3_byte")], stride[("v3d", "seq_memcpy")]
    _bar(ax, 0, BAR_HEIGHT, gpu_byte, GPU_MEM_COLOR, x_max, _two(gpu_byte.median))
    _bar(ax, 1 - PAIR_HEIGHT / 2, PAIR_HEIGHT, gpu_tr, GPU_MEM_COLOR, x_max, _two(gpu_tr.median))
    _bar(ax, 1 + PAIR_HEIGHT / 2, PAIR_HEIGHT, cached_tr, CACHED_MEM_COLOR, x_max,
         f"{_two(cached_tr.median)} (캐시 dma-heap {_two(heap_tr.median)})")
    _bar(ax, 2, BAR_HEIGHT, gpu_copy, GPU_MEM_COLOR, x_max, _two(gpu_copy.median))
    _bar(ax, 3, BAR_HEIGHT, writes["seq_byte"], GPU_MEM_COLOR, x_max, _two(writes["seq_byte"].median))
    _bar(ax, 4, BAR_HEIGHT, writes["memcpy"], GPU_MEM_COLOR, x_max, _two(writes["memcpy"].median))
    _bar(ax, 5, BAR_HEIGHT, gpu_ms, GPU_MEM_COLOR, x_max, f"{_two(gpu_ms.median)} (대역폭 환산)", hatch=CONVERTED_HATCH)

    ax.set_yticks(range(6), ["CPU 1바이트 읽기", "CPU 전치 루프\n(1바이트, stride 3)", "CPU memcpy 읽기", "CPU 1바이트 쓰기",
                             "CPU memcpy 쓰기", f"GPU 읽기\n({ps.fmt_half_up(gpu_gbps.median, 2)} GB/s에서 환산)"])
    ax.set_ylim(5.6, -0.6)
    ax.set_xlim(0, x_max)
    ax.xaxis.set_major_locator(MultipleLocator(X_TICK_STEP))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"{BUFFER_BYTES:,} B 접근 시간 (ms)")
    for y in (2.5, 4.5):                 # CPU read / CPU write / GPU read
        ax.axhline(y, color=ps.REFERENCE_COLOR, lw=0.5, ls=(0, (3, 2)))

    date_x = ax.get_yaxis_transform()    # x in axes coordinates, y in data coordinates
    for y, date in ((0, read_date), (1, read_date), (2, read_date), (3, asym_date), (4, asym_date), (5, asym_date)):
        ax.text(1.0, y, date, transform=date_x, ha="right", va="center", color=ps.REFERENCE_COLOR)
    ax.text(1.0, -0.6, "측정일", transform=date_x, ha="right", va="bottom", color=ps.REFERENCE_COLOR)

    handles = [Patch(fc=GPU_MEM_COLOR, ec=ps.EDGE_COLOR, lw=0.4, label=GPU_MEM_LABEL),
               Patch(fc=CACHED_MEM_COLOR, ec=ps.EDGE_COLOR, lw=0.4, label=CACHED_MEM_LABEL)]
    ax.legend(handles=handles, loc="center right", bbox_to_anchor=(0.84, 0.3), handlelength=1.6)

    conditions = {(s["governor"], s["v3d"]) for s in sessions.values()}
    if len(conditions) != 1:
        raise ValueError(f"the two sessions differ in governor or V3D clock, draw them separately: {sessions}")
    governor, v3d = conditions.pop()
    fig.text(NOTE_X, NOTE_Y, f"두 회차 공통: 거버너 {governor}, V3D {v3d} MHz, NPU·표시 미사용, CPU 측정 코어 3 고정. "
             "막대는 실행별 중앙값의 3회 중앙값, 수염은 최소~최대.\n"
             "GPU 읽기는 uvec4 접근, 디스패치 형태 256×3, 디스패치 고정비 포함 값이며 형태에 따라 크게 달라진다.",
             ha="left", va="bottom", linespacing=1.2)

    lines = [f"{read_date} stride_read governor {sessions['stride']['governor']}, V3D {sessions['stride']['v3d']} MHz",
             f"{asym_date} memory_asymmetry governor {sessions['asym']['governor']}, V3D {sessions['asym']['v3d']} MHz"]
    for name, v in (("CPU seq_byte read v3d", gpu_byte), ("CPU transpose v3d", gpu_tr), ("CPU transpose malloc", cached_tr),
                    ("CPU transpose heap", heap_tr), ("CPU memcpy read v3d", gpu_copy),
                    ("CPU seq_byte write v3d", writes["seq_byte"]), ("CPU memcpy write v3d", writes["memcpy"])):
        lines.append(f"{name}: {v.median:.3f} [{v.low:.3f}..{v.high:.3f}] -> {_two(v.median)}")
    lines.append(f"GPU read_v4 v3d raw {gpu_gbps.median:.3f} GB/s [{gpu_gbps.low:.3f}..{gpu_gbps.high:.3f}] -> "
                 f"{gpu_ms.median:.4f} ms -> {_two(gpu_ms.median)}; net {gpu_net:.3f} GB/s -> "
                 f"{_gbps_to_ms(gpu_net):.4f} ms (caption only)")
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [STRIDE_DIR / "evidence" / "stride_run1.txt", STRIDE_DIR / "env_bench.txt", MEDIANS,
                     ASYM_DIR / "raw" / "cpuwrite_run1" / "stdout.log", ASYM_DIR / "raw" / "bw_r256x3_run1" / "stdout.log",
                     ASYM_DIR / "env_bench.txt"], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
