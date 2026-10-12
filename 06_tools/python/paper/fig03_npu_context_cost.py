"""Figure 3: per-context fixed cost a and per-frame cost b of the batch span (yolov11m.hef, 4 contexts).

Source: hef_context_profile/evidence/breakdown.json `fits.contexts` — per context, span(B) = a + b*B fitted over
9 runs (B = 1/2/8 x 3) of `hailortcli run2 ... measure-fw-actions` firmware timestamps (PCIe Gen3 x1).
Two stacked bars: the fixed part a (per batch) and the per-frame part b, each split into dynamic_0..3.
The JSON `fps` fields and the `fits.totals.fps_period` fit derived from them are NOT used (do-not-cite,
hef_context_profile/summary.md §4); the total shown is the sum of the context fits, cross-checked against
`fits.totals.batch_span` in the audit output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import paper_style as ps

STEM = "fig03_npu_context_cost"
SRC = ps.RESULTS / "hef_context_profile" / "evidence" / "breakdown.json"
CONTEXTS = ("dynamic_0", "dynamic_1", "dynamic_2", "dynamic_3")


def load() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """(context -> {a, b, rms, n}, batch_span total fit)."""
    fits = ps.read_json(SRC)["fits"]
    ctx = {c: fits["contexts"][c] for c in CONTEXTS}
    return ctx, fits["totals"]["batch_span"]


def main(out_dir: str | Path | None = None) -> list[Path]:
    ctx, span = load()
    bars = (("a", "고정 a\n(배치당)"), ("b", "프레임당 b\n(× B)"))
    fig, ax = ps.new_figure("single", 5.2)
    colors = dict(zip(CONTEXTS, (ps.PALETTE[0], ps.PALETTE[1], ps.PALETTE[2], ps.PALETTE[4])))
    for row, (key, _) in enumerate(bars):
        left = 0.0
        for c in CONTEXTS:
            w = ctx[c][key]
            ax.barh(row, w, left=left, height=0.55, color=colors[c], edgecolor="white", linewidth=0.6,
                    label=c if row == 0 else None)
            if w >= 1.5:
                ax.text(left + w / 2, row, ps.fmt_half_up(w, 2), ha="center", va="center", fontsize=6.5,
                        color="white" if c != "dynamic_1" else "black")
            else:   # too narrow for an inside label: put it just above the bar
                ax.text(left + w / 2, row - 0.31, ps.fmt_half_up(w, 2), ha="center", va="bottom", fontsize=6.5)
            left += w
        ax.text(left + 0.4, row, f"합 {ps.fmt_half_up(left, 2)} ms", va="center", fontsize=7)
    ax.set_yticks(range(len(bars)), [lab for _, lab in bars])
    ax.invert_yaxis()
    ax.set_xlabel("시간 (ms)")
    ax.set_xlim(0, 31)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncols=4, handlelength=1.0, columnspacing=0.8)
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)

    sum_a = sum(ctx[c]["a"] for c in CONTEXTS)
    sum_b = sum(ctx[c]["b"] for c in CONTEXTS)
    lines = [f"{c}: a {ctx[c]['a']:.2f} ms ({100 * ctx[c]['a'] / span['a']:.1f}% of batch_span a), b {ctx[c]['b']:.2f} ms, "
             f"rms {ctx[c]['rms']:.3f}, n {ctx[c]['n']}" for c in CONTEXTS]
    lines.append(f"sum of contexts: a {sum_a:.2f} ms, b {sum_b:.2f} ms")
    lines.append(f"cross-check fits.totals.batch_span: a {span['a']:.2f}, b {span['b']:.2f}, rms {span['rms']:.2f} "
                 f"(difference in a = context transitions {span['a'] - sum_a:.3f} ms)")
    lines.append("not used: JSON fps, fits.totals.fps_period (do-not-cite)")
    ps.report(STEM, [SRC], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
