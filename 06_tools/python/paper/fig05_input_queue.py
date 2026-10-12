"""Figure 5: leaky input queue size N vs arrival-based latency (e2e_arr) and drop rate, b1/s1 and b2/s4.

Source: input_pacing/results_raw.csv (one row per run; settings src_b1s1_q{1,2,3} and src_b2s4_q{1,2,3},
3 runs each, `--input-pace source`, display off). Plotted per N: median over the 3 runs with the min-max range of
  e2e_arr_mean, e2e_arr_p90 (ms, arrival time t0 + k/30 s -> result complete) and drop_rate (dropped / arrived).
The asap control rows have no e2e_arr (their e2e starts at decode end) and are not drawn — different definition.
Runs must have ok = 1, pace_ok = 1 and no hailort warnings/errors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import paper_style as ps

STEM = "fig05_input_queue"
SRC = ps.RESULTS / "input_pacing" / "results_raw.csv"
CONFIGS = (("b1s1", "b1/s1 (O2)", ps.PALETTE[0]), ("b2s4", "b2/s4", ps.PALETTE[1]))
QUEUES = (1, 2, 3)
METRICS = ("e2e_arr_mean", "e2e_arr_p90", "drop_rate")


def load() -> dict[str, dict[int, dict[str, list[float]]]]:
    """config -> N -> metric -> per-run values."""
    out: dict[str, dict[int, dict[str, list[float]]]] = {}
    for r in ps.read_csv(SRC):
        cfg, _, q = r["label"].removeprefix("src_").partition("_q")
        if not r["label"].startswith("src_") or cfg not in dict((c, 0) for c, _, _ in CONFIGS):
            continue
        valid = r["ok"] == "1" and r["pace_ok"] == "1" and r["hailort_warnings"] == "0" and r["hailort_errors"] == "0"
        if not valid:
            raise ValueError(f"{ps.rel(SRC)}: run {r['label']} run{r['run']} is not valid")
        d = out.setdefault(cfg, {}).setdefault(int(q), {m: [] for m in METRICS})
        for m in METRICS:
            d[m].append(float(r[m]))
    return out


def main(out_dir: str | Path | None = None) -> list[Path]:
    data = load()
    fig, (ax1, ax2) = ps.new_figure("double", 6.0, ncols=2)
    lines = []
    for cfg, label, col in CONFIGS:
        d = data[cfg]
        ps.errorbar_median(ax1, QUEUES, [d[n]["e2e_arr_mean"] for n in QUEUES], fmt="o-", color=col,
                           label=f"{label} 평균")
        ps.errorbar_median(ax1, QUEUES, [d[n]["e2e_arr_p90"] for n in QUEUES], fmt="s--", color=col, mfc="white",
                           label=f"{label} p90")
        ps.errorbar_median(ax2, QUEUES, [[100 * v for v in d[n]["drop_rate"]] for n in QUEUES], fmt="o-",
                           color=col, label=label)
        for n in QUEUES:
            mean = ps.median_min_max(d[n]["e2e_arr_mean"])
            p90 = ps.median_min_max(d[n]["e2e_arr_p90"])
            drop = ps.median_min_max(d[n]["drop_rate"])
            if cfg == "b1s1":   # b2/s4 is flat: labelled once at its right end below
                # offsets keep every label clear of the lines; N=2 sits between the lines with a leader
                off, ha, arrow = {QUEUES[0]: ((-6, 5), "right", None), QUEUES[-1]: ((8, -10), "left", None)}.get(
                    n, ((0, -24), "center", {"arrowstyle": "-", "color": col, "lw": 0.5, "shrinkA": 1, "shrinkB": 3}))
                ax1.annotate(ps.fmt_half_up(mean[0]), (n, mean[0]), textcoords="offset points", xytext=off,
                             ha=ha, va="center" if arrow else "baseline", fontsize=6.5, arrowprops=arrow)
            ax2.annotate(f"{ps.fmt_half_up(100 * drop[0])}%", (n, 100 * drop[0]), textcoords="offset points", xytext=(0, 5),
                         ha="center", fontsize=6.5)
            lines.append(f"{cfg} N={n}: runs {len(d[n]['e2e_arr_mean'])}, e2e_arr mean {mean[0]:.2f} "
                         f"({mean[1]:.2f}..{mean[2]:.2f}), p90 {p90[0]:.2f} ({p90[1]:.2f}..{p90[2]:.2f}), "
                         f"drop {100 * drop[0]:.2f}% ({100 * drop[1]:.2f}..{100 * drop[2]:.2f})")
    b1 = [ps.median_min_max(data["b1s1"][n]["e2e_arr_mean"])[0] for n in QUEUES]
    step = (b1[-1] - b1[0]) / (QUEUES[-1] - QUEUES[0])
    ax1.text(0.75, 12, f"b1/s1 평균:\nN 한 단계마다 +{ps.fmt_half_up(step)} ms\n(1/30 s = 33.3 ms)", fontsize=6.5,
             color=ps.PALETTE[0])
    b2 = [ps.median_min_max(data["b2s4"][n]["e2e_arr_mean"])[0] for n in QUEUES]
    ax1.annotate(f"{ps.fmt_half_up(min(b2))}–{ps.fmt_half_up(max(b2))}", (QUEUES[-1], b2[-1]), textcoords="offset points",
                 xytext=(8, -3), ha="left", va="center", fontsize=6.5)
    lines.append(f"b1s1 mean slope over N: +{step:.2f} ms per step (frame interval {1000 / 30:.2f} ms)")
    for ax in (ax1, ax2):
        ax.set_xticks(QUEUES)
        ax.set_xlim(0.7, 3.65)
        ax.set_xlabel("입력 큐 크기 N (프레임)")
    ax1.set_ylabel("도착 기준 지연 e2e_arr (ms)")
    ax1.set_ylim(0, 150)
    ax1.set_title("(a) 도착 → 결과 지연", loc="left")
    ax1.legend(loc="lower right", ncols=1)
    ax2.set_ylabel("버린 프레임 비율 (%)")
    ax2.set_ylim(-1, 25)
    ax2.set_title("(b) 입력 버림률", loc="left")
    ax2.legend(loc="center right")
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)
    ps.report(STEM, [SRC], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
