"""Figure 2: YOLO11m blocks on the 4 HEF contexts (a) and per-context fixed cost a / per-frame cost b (b).

(a) Source: hef_context_profile/evidence/context_layers.json (06_tools/python/hef_context_layers.py). Everything drawn comes
from the JSON: block type, yaml `from` (main flow, Concat skips, Detect inputs) and stride comments from `blocks` (block
table of ultralytics yolo11.yaml); the per-block context list from `blocks[].contexts`; how a block that spans two contexts
is split (model.6, model.23) from its ONNX node count per context (`contexts[].module_paths`); the Detect heads from the
cv2.i/cv3.i paths; the 6 chip outputs from the BOUNDARY D2H `schema_edges` (shape, stride); the tensor shapes on edges that
cross a context from the INTER_CONTEXT D2H `schema_edges`. The context lists are position-inferred from the HEF compiler
metadata and cross-checked with the schema edge layers and fw-actions counts; every `checks` flag must be true.
(b) Source: hef_context_profile/evidence/breakdown.json `fits.contexts` — per context, span(B) = a + b*B fitted over
9 runs (B = 1/2/8 x 3) of `hailortcli run2 ... measure-fw-actions` firmware timestamps (PCIe Gen3 x1).
Two stacked bars: the fixed part a (per batch) and the per-frame part b, each split into dynamic_0..3.
The JSON `fps` fields and the `fits.totals.fps_period` fit derived from them are NOT used (do-not-cite,
hef_context_profile/summary.md §4); the total shown is the sum of the context fits, cross-checked against
`fits.totals.batch_span` in the audit output.
"""
from __future__ import annotations

import ast
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

import paper_style as ps

STEM = "fig02_npu_context_cost"
SRC = ps.RESULTS / "hef_context_profile" / "evidence" / "breakdown.json"
LAYERS_SRC = ps.RESULTS / "hef_context_profile" / "evidence" / "context_layers.json"
CONTEXTS = ("dynamic_0", "dynamic_1", "dynamic_2", "dynamic_3")
CTX_COLORS = dict(zip(CONTEXTS, (ps.PALETTE[0], ps.PALETTE[1], ps.PALETTE[2], ps.PALETTE[4])))
CTX_TEXT = {c: ("black" if c == "dynamic_1" else "white") for c in CONTEXTS}   # label color on a context fill
DETECT = "Detect"
LINE_COLOR = ps.EDGE_COLOR
SKIP_COLOR = "#6E6E6E"
REF = ps.REFERENCE_COLOR

# ─── Layout of panel (a), in cm (drawing geometry only, no data) ───────────────────────────────────────────────────
FIG_W = ps.DOUBLE_COL_CM
A_H = 7.9                    # panel (a) height
B_H = 3.9                    # panel (b) height
PITCH = 1.42                 # column pitch of the block boxes
BOX_W = 1.16
BOX_H = 0.98
X_LEFT = 0.25                # left edge of column 0
Y1T = A_H - 1.05             # lane 1 (backbone) top
Y2T = Y1T - BOX_H - 1.5      # lane 2 (neck) top; the 1.5 cm between is the routing channel
LEVEL_SKIP_UP = Y1T - BOX_H - 0.35   # 6 -> 12 runs here
LEVEL_WRAP = Y1T - BOX_H - 0.75      # 10 -> 11 (lane wrap) runs here
LEVEL_SKIP_DN = Y2T + 0.32           # 13 -> 18 runs here
HEAD_T = Y2T - BOX_H - 0.62  # Detect head boxes
HEAD_H = 0.78
HEAD_W = 1.8
CHIP_LINE = 1.16             # chip | host boundary
HOST_T = 0.78
HOST_B = 0.1


@dataclass
class Block:
    index: int
    module: str
    srcs: list[int]                      # resolved yaml `from` (-1 = network input)
    comment: str
    shares: list[tuple[str, int]]        # (context, ONNX nodes of this block in that context), in context order
    stride: int = 0

    @property
    def out_ctx(self) -> str:
        return self.shares[-1][0]

    @property
    def in_ctx(self) -> str:
        return self.shares[0][0]


@dataclass
class Head:
    level: int                           # Detect input i (0 = P3)
    src: int                             # block feeding this head
    shares: list[tuple[str, int]]
    stride: int = 0
    outputs: list[dict] = field(default_factory=list)   # BOUNDARY D2H edges: branch, shape, ctx, stride


@dataclass
class Structure:
    blocks: list[Block]
    heads: list[Head]
    input_shape: list[int]
    cross: dict[tuple[int, int], list[int]]   # (src block, dst block) -> shape of the inter-context tensor
    contexts: dict[str, dict]


def load() -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """(context -> {a, b, rms, n}, batch_span total fit)."""
    fits = ps.read_json(SRC)["fits"]
    ctx = {c: fits["contexts"][c] for c in CONTEXTS}
    return ctx, fits["totals"]["batch_span"]


def _resolve_from(raw: str, index: int) -> list[int]:
    v = ast.literal_eval(raw)
    return [index + s if s < 0 else s for s in (v if isinstance(v, list) else [v])]


def _stride_of(blk: dict, srcs: list[int], strides: dict[int, int]) -> int:
    """Output stride from the yaml args: Conv [c, k, s], nn.Upsample [None, scale, mode], other blocks keep it."""
    s_in = strides[srcs[0]] if srcs[0] >= 0 else 1
    match blk["module"]:
        case "Conv":
            args = ast.literal_eval(blk["yaml_args"])
            return s_in * (args[2] if len(args) > 2 else 1)
        case "nn.Upsample":
            return s_in // ast.literal_eval(blk["yaml_args"])[1]
        case _:
            return s_in


def load_structure() -> Structure:
    """Blocks, Detect heads, chip outputs and inter-context tensors from context_layers.json (see module docstring)."""
    d = ps.read_json(LAYERS_SRC)
    bad = [k for k, v in d["checks"].items() if isinstance(v, bool) and not v]
    if bad:
        raise SystemExit(f"{ps.rel(LAYERS_SRC)}: checks failed: {bad}")
    ctxs = {c["fw_context"]: c for c in d["contexts"]}
    if tuple(sorted(ctxs)) != CONTEXTS:
        raise SystemExit(f"{ps.rel(LAYERS_SRC)}: unexpected contexts {sorted(ctxs)}")
    by_index = {c["index"]: c["fw_context"] for c in d["contexts"]}

    blocks: list[Block] = []
    strides: dict[int, int] = {}
    for blk in d["blocks"]:
        i = blk["index"]
        srcs = _resolve_from(blk["yaml_from"], i)
        shares = [(c, len(ctxs[c]["module_paths"][str(i)])) for c in CONTEXTS if str(i) in ctxs[c]["module_paths"]]
        if [by_index[k] for k in blk["contexts"]] != [c for c, _ in shares]:
            raise SystemExit(f"model.{i}: block contexts {blk['contexts']} disagree with module_paths {shares}")
        b = Block(i, blk["module"], srcs, blk["yaml_comment"], shares)
        if b.module != DETECT:
            b.stride = strides[i] = _stride_of(blk, srcs, strides)
            m = re.match(r"P(\d)/(\d+)", b.comment)
            if m and (int(m.group(2)) != b.stride or 2 ** int(m.group(1)) != b.stride):
                raise SystemExit(f"model.{i}: derived stride {b.stride} disagrees with yaml comment {b.comment!r}")
        blocks.append(b)

    det = next(b for b in blocks if b.module == DETECT)
    heads: list[Head] = []
    for lvl, src in enumerate(det.srcs):
        pre = (f"cv2.{lvl}/", f"cv3.{lvl}/")
        shares = [(c, n) for c in CONTEXTS
                  if (n := sum(p.startswith(pre) for p in ctxs[c]["module_paths"].get(str(det.index), [])))]
        heads.append(Head(lvl, src, shares, stride=strides[src]))

    input_shape: list[int] = []
    cross: dict[tuple[int, int], list[int]] = {}
    for c in CONTEXTS:
        for e in ctxs[c]["schema_edges"]:
            if e["connection"] != "BOUNDARY":
                continue
            if e["direction"] == "H2D":
                input_shape = e["shape"]
                continue
            m = re.match(rf"/model\.{det.index}/cv([23])\.(\d)/", e["original_names"][0])
            if not m:
                raise SystemExit(f"unexpected chip output {e['name']}: {e['original_names']}")
            h = heads[int(m.group(2))]
            if e["stride"] != h.stride:
                raise SystemExit(f"{e['name']}: stride {e['stride']} but head P{h.level + 3} input stride {h.stride}")
            h.outputs.append({"branch": f"cv{m.group(1)}", "shape": e["shape"], "ctx": c, "stride": e["stride"]})
    for h in heads:
        h.outputs.sort(key=lambda o: o["branch"])

    # tensor shape on every data edge whose ends sit in different contexts (sent D2H by the producing context);
    # a Detect input goes to its own head, which may sit in another context than the rest of model.23
    links = [(s, b.index, b.in_ctx) for b in blocks if b.module != DETECT for s in b.srcs if s >= 0]
    links += [(h.src, det.index, h.shares[0][0]) for h in heads]
    for s, t, dst_ctx in links:
        src = blocks[s]
        if src.out_ctx == dst_ctx:
            continue
        dst_idx = next(k for k, v in by_index.items() if v == dst_ctx)
        edge = next((e for e in ctxs[src.out_ctx]["schema_edges"]
                     if e["direction"] == "D2H" and e["connection"] == "INTER_CONTEXT"
                     and dst_idx in e["connected_contexts"]
                     and any(n.startswith(f"/model.{s}/") for n in e["original_names"])), None)
        if edge is None:
            raise SystemExit(f"no INTER_CONTEXT edge for model.{s} ({src.out_ctx}) -> model.{t} ({dst_ctx})")
        cross[(s, t)] = edge["shape"]
    return Structure(blocks, heads, input_shape, cross, ctxs)


# ─── Panel (a) ──────────────────────────────────────────────────────────────────────────────────────────────────────
def _shape(s: list[int]) -> str:
    return "×".join(str(v) for v in s)


def _level(stride: int) -> str:
    return f"P{int(math.log2(stride))}/{stride}"


def _col_x(col: float) -> float:
    return X_LEFT + BOX_W / 2 + col * PITCH


def _segments(x_left: float, width: float, shares: list[tuple[str, int]]) -> list[tuple[str, float, float]]:
    """(context, x0, x1) pieces of a box, widths proportional to the node shares."""
    total = sum(n for _, n in shares)
    out, x = [], x_left
    for c, n in shares:
        w = width * n / total
        out.append((c, x, x + w))
        x += w
    return out


def _box(ax: Axes, x_left: float, y_bottom: float, width: float, height: float, shares: list[tuple[str, int]],
         label: str) -> list[tuple[str, float, float]]:
    """Box filled with its context color(s); a block over two contexts is split by node share (black label)."""
    segs = _segments(x_left, width, shares)
    for c, x0, x1 in segs:
        ax.add_patch(Rectangle((x0, y_bottom), x1 - x0, height, facecolor=CTX_COLORS[c], edgecolor="none", zorder=3))
    for _, x0, _ in segs[1:]:      # split marker between two contexts
        ax.plot([x0, x0], [y_bottom, y_bottom + height], color="white", lw=1.2, zorder=4, solid_capstyle="butt")
    ax.add_patch(Rectangle((x_left, y_bottom), width, height, facecolor="none", edgecolor=LINE_COLOR, lw=0.6, zorder=5))
    split = len(segs) > 1
    ax.text(x_left + width / 2, y_bottom + height / 2, label, ha="center", va="center", fontsize=6.8,
            color="black" if split else CTX_TEXT[shares[0][0]], linespacing=1.2, zorder=6)
    return segs


def _poly_arrow(ax: Axes, pts: list[tuple[float, float]], color: str = LINE_COLOR, lw: float = 0.8,
                z: float = 2) -> None:
    """Polyline whose last segment ends in an arrowhead."""
    if len(pts) > 2:
        xs, ys = zip(*pts[:-1])
        ax.plot(xs, ys, color=color, lw=lw, zorder=z, solid_joinstyle="miter", solid_capstyle="butt")
    ax.annotate("", xy=pts[-1], xytext=pts[-2], zorder=z,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0, mutation_scale=6))


class _SplitHandle:
    def __init__(self, colors: tuple[str, str]) -> None:
        self.colors = colors


class _SplitHandler(HandlerBase):
    """Legend key drawn as a box split into two context colors."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        half = width / 2
        arts = [Rectangle((-xdescent + k * half, -ydescent), half, height, facecolor=c, edgecolor="none", transform=trans)
                for k, c in enumerate(orig_handle.colors)]
        arts.append(Rectangle((-xdescent, -ydescent), width, height, facecolor="none", edgecolor=LINE_COLOR, lw=0.5,
                              transform=trans))
        return arts


def _ctx_desc(st: Structure, c: str) -> str:
    """Which blocks a context holds, for the legend (full blocks as ranges, split blocks as '일부')."""
    full = [b.index for b in st.blocks if b.module != DETECT and [x for x, _ in b.shares] == [c]]
    part = [b.index for b in st.blocks if b.module != DETECT and len(b.shares) > 1 and c in [x for x, _ in b.shares]]
    runs: list[str] = []
    for i in full:
        if runs and int(runs[-1].split("–")[-1]) == i - 1:
            runs[-1] = f"{runs[-1].split('–')[0]}–{i}"
        else:
            runs.append(str(i))
    items = [f"model.{r}" for r in runs] + [f"model.{i} 일부" for i in part]
    heads = [f"P{h.level + 3}" + ("" if len(h.shares) == 1 else " 일부") for h in st.heads if c in [x for x, _ in h.shares]]
    if heads:
        items.append("Detect " + "·".join(heads))
    return ", ".join(items)


def draw_structure(ax: Axes, st: Structure) -> None:
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, A_H)
    ax.set_axis_off()
    blocks = [b for b in st.blocks if b.module != DETECT]
    det = next(b for b in st.blocks if b.module == DETECT)
    neck0 = next(b.index for b in blocks if b.module == "nn.Upsample")   # first neck block (11)
    pos: dict[int, tuple[float, float]] = {}                              # block -> (center x, top y)
    for b in blocks:
        col, top = (b.index, Y1T) if b.index < neck0 else (b.index - neck0, Y2T)
        x = _col_x(col)
        pos[b.index] = (x, top)
        _box(ax, x - BOX_W / 2, top - BOX_H, BOX_W, BOX_H, b.shares,
             f"{b.index}\n{b.module.removeprefix('nn.')}\n{_level(b.stride)}")

    def bottom(i: int) -> float:
        return pos[i][1] - BOX_H

    def right(i: int) -> float:
        return pos[i][0] + BOX_W / 2

    def left(i: int) -> float:
        return pos[i][0] - BOX_W / 2

    def mid_y(i: int) -> float:
        return pos[i][1] - BOX_H / 2

    # main flow (yaml from -1) inside a lane, and the lane wrap 10 -> 11
    last_backbone = neck0 - 1
    for b in blocks:
        s = b.srcs[0]
        if s < 0:
            continue
        if b.index == neck0:
            gap_x = right(s) + (PITCH - BOX_W) / 2 + 0.05
            _poly_arrow(ax, [(right(s), mid_y(s)), (gap_x, mid_y(s)), (gap_x, LEVEL_WRAP),
                             (pos[b.index][0] - 0.22, LEVEL_WRAP), (pos[b.index][0] - 0.22, pos[b.index][1])], lw=0.9)
        else:
            _poly_arrow(ax, [(right(s), mid_y(s)), (left(b.index), mid_y(b.index))], lw=0.9)

    # skips: second Concat input
    for b in blocks:
        if len(b.srcs) < 2:
            continue
        s = b.srcs[1]
        xs, xd = pos[s][0], pos[b.index][0]
        if s <= last_backbone and abs(xs - xd) < 1e-6:          # straight down (4 -> 15, 10 -> 21)
            _poly_arrow(ax, [(xs, bottom(s)), (xd, pos[b.index][1])], color=SKIP_COLOR, lw=0.6)
        elif s <= last_backbone:                                # backbone -> neck, routed (6 -> 12)
            _poly_arrow(ax, [(xs, bottom(s)), (xs, LEVEL_SKIP_UP), (xd + 0.22, LEVEL_SKIP_UP),
                             (xd + 0.22, pos[b.index][1])], color=SKIP_COLOR, lw=0.6)
        else:                                                   # inside the neck (13 -> 18)
            _poly_arrow(ax, [(xs, pos[s][1]), (xs, LEVEL_SKIP_DN), (xd, LEVEL_SKIP_DN), (xd, pos[b.index][1])],
                        color=SKIP_COLOR, lw=0.6)

    # tensor shapes on edges that cross a context boundary
    lab = dict(fontsize=6.3, color=LINE_COLOR, zorder=7)
    y_mid = (LEVEL_SKIP_UP + LEVEL_WRAP) / 2
    for (s, t), shape in sorted(st.cross.items()):
        txt = _shape(shape)
        xs, xt = pos[s][0], pos[t][0]
        if t == neck0:                                          # lane wrap: label it unless a straight skip shares it
            if not any(k[0] == s and k[1] != t and abs(pos[k[1]][0] - xs) < 1e-6 for k in st.cross):
                ax.text((right(s) + _col_x(neck0)) / 2, LEVEL_WRAP + 0.04, txt, ha="center", va="bottom", **lab)
        elif s <= last_backbone < t and abs(xs - xt) < 1e-6:    # straight skip; the lane wrap runs on its right
            wrap_side = s == last_backbone
            ax.text(xs + (-0.06 if wrap_side else 0.06), y_mid, txt, ha="right" if wrap_side else "left",
                    va="center", **lab)
        elif s <= last_backbone < t:                            # routed skip: label just right of its descent
            ax.text(xt + 0.3, LEVEL_SKIP_UP + 0.04, txt, ha="left", va="bottom", **lab)
        else:                                                   # neighbours in the neck lane (21 -> 22)
            ax.text((right(s) + left(t)) / 2 + 0.1, bottom(s) - 0.08, txt, ha="right", va="top", **lab)

    # Detect heads under their input blocks, then the chip outputs
    head_pos: dict[int, list[tuple[str, float, float]]] = {}
    for h in st.heads:
        x = pos[h.src][0]
        segs = _box(ax, x - HEAD_W / 2, HEAD_T - HEAD_H, HEAD_W, HEAD_H, h.shares,
                    f"P{h.level + 3}/{h.stride} 머리\ncv2.{h.level} · cv3.{h.level}")
        head_pos[h.level] = segs
        _poly_arrow(ax, [(x, bottom(h.src)), (x, HEAD_T)], color=SKIP_COLOR, lw=0.6)
    ax.text(_col_x(st.heads[0].src - neck0) - HEAD_W / 2 - 0.12, HEAD_T - HEAD_H / 2,
            f"Detect (model.{det.index})\n머리 {len(st.heads)}개", ha="right", va="center", fontsize=7)

    for h in st.heads:
        xs_out = []
        for k, o in enumerate(h.outputs):
            c0, x0, x1 = next(sg for sg in head_pos[h.level] if sg[0] == o["ctx"])
            n_here = sum(1 for oo in h.outputs if oo["ctx"] == o["ctx"])
            j = sum(1 for oo in h.outputs[:k] if oo["ctx"] == o["ctx"])
            x = x0 + (x1 - x0) * (j + 1) / (n_here + 1)
            xs_out.append(x)
            _poly_arrow(ax, [(x, HEAD_T - HEAD_H), (x, HOST_T)], color=CTX_COLORS[c0], lw=1.1, z=4)
        ax.text(min(xs_out) - 0.08, (HEAD_T - HEAD_H + CHIP_LINE) / 2 + 0.02,
                "\n".join(f"{o['branch']} {_shape(o['shape'])}" for o in h.outputs),
                ha="right", va="center", linespacing=1.15, **lab)

    # chip | host boundary and the host post-processing box
    x_from = _col_x(st.heads[0].src - neck0) - HEAD_W / 2 - 0.9
    x_note = pos[st.heads[0].src][0] + HEAD_W / 2 + 0.15      # free span right of the first head
    ax.plot([x_from, FIG_W - 0.2], [CHIP_LINE, CHIP_LINE], color=REF, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.text(x_note, CHIP_LINE + 0.05, "▲ Hailo-8 칩", ha="left", va="bottom", fontsize=6.5, color=REF)
    ax.text(x_note, CHIP_LINE - 0.05, "▼ 호스트", ha="left", va="top", fontsize=6.5, color=REF)
    ax.add_patch(Rectangle((x_from, HOST_B), FIG_W - 0.2 - x_from, HOST_T - HOST_B, facecolor="white",
                           edgecolor=LINE_COLOR, lw=0.6, zorder=3))
    per_ctx = {c: [o for h in st.heads for o in h.outputs if o["ctx"] == c] for c in CONTEXTS}
    parts = [f"{c} {len(v)}개 (stride {'·'.join(str(s) for s in sorted({o['stride'] for o in v}))})"
             for c, v in per_ctx.items() if v]
    n_out = sum(len(v) for v in per_ctx.values())
    ax.text((x_from + FIG_W - 0.2) / 2, (HOST_T + HOST_B) / 2,
            f"호스트 후처리 (디코드 · NMS)\n칩 밖 출력 {n_out}개 — " + ", ".join(parts),
            ha="center", va="center", fontsize=7, linespacing=1.25, zorder=6)

    # input, lane names
    x_in = pos[0][0]
    ax.annotate(f"입력 {_shape(st.input_shape)}", xy=(x_in, Y1T), xytext=(x_in - BOX_W / 2, Y1T + 0.42),
                ha="left", va="center", fontsize=7,
                arrowprops=dict(arrowstyle="-|>", color=LINE_COLOR, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=6,
                                relpos=(0.08, 0.0)))
    ax.text(_col_x(neck0) + 0.15, mid_y(0), f"백본\nmodel.0–{last_backbone}", ha="center", va="center", fontsize=7)
    ax.text(X_LEFT, Y2T - BOX_H - 0.12, f"넥 model.{neck0}–{det.index - 1}", ha="left", va="top", fontsize=7)

    # legends: contexts (lower left) and drawing marks (top right)
    ctx_handles = [Patch(facecolor=CTX_COLORS[c], edgecolor="none", label=f"{c}: {_ctx_desc(st, c)}") for c in CONTEXTS]
    leg = ax.legend(handles=ctx_handles, loc="lower left", bbox_to_anchor=(X_LEFT - 0.1, HOST_B - 0.05),
                    bbox_transform=ax.transData, fontsize=6.8, handlelength=1.4, handleheight=0.9,
                    labelspacing=0.35, borderaxespad=0, title="HEF 컨텍스트 (그림 (b)와 같은 색)",
                    title_fontsize=7, alignment="left")
    ax.add_artist(leg)
    split = _SplitHandle((CTX_COLORS[CONTEXTS[0]], CTX_COLORS[CONTEXTS[1]]))
    marks = [split,
             Line2D([], [], color=LINE_COLOR, lw=0.9),
             Line2D([], [], color=SKIP_COLOR, lw=0.6),
             Line2D([], [], color="none")]
    labels = ["두 컨텍스트에 걸친 블록 (칠 폭 = 노드 수 비)",
              "주 흐름",
              "skip (Concat · Detect 입력)",
              "H×W×C: 컨텍스트 경계를 넘는 텐서"]
    ax.legend(marks, labels, handler_map={_SplitHandle: _SplitHandler()}, loc="upper right",
              bbox_to_anchor=(FIG_W - 0.1, A_H - 0.02), bbox_transform=ax.transData, ncols=2, fontsize=6.8,
              handlelength=1.6, columnspacing=1.2, borderaxespad=0, handletextpad=0.5, labelspacing=0.3)



# ─── Panel (b) ──────────────────────────────────────────────────────────────────────────────────────────────────────
def draw_cost(ax: Axes, ctx: dict[str, dict[str, float]]) -> None:
    bars = (("a", "고정 a\n(배치당)"), ("b", "프레임당 b\n(× B)"))
    for row, (key, _) in enumerate(bars):
        left = 0.0
        for c in CONTEXTS:
            w = ctx[c][key]
            ax.barh(row, w, left=left, height=0.55, color=CTX_COLORS[c], edgecolor="white", linewidth=0.6)
            if w >= 1.5:
                ax.text(left + w / 2, row, ps.fmt_half_up(w, 2), ha="center", va="center", fontsize=6.5,
                        color=CTX_TEXT[c])
            else:   # too narrow for an inside label: put it just above the bar
                ax.text(left + w / 2, row - 0.31, ps.fmt_half_up(w, 2), ha="center", va="bottom", fontsize=6.5)
            left += w
        ax.text(left + 0.3, row, f"합 {ps.fmt_half_up(left, 2)} ms", va="center", fontsize=7)
    ax.set_yticks(range(len(bars)), [lab for _, lab in bars])
    ax.invert_yaxis()
    ax.set_xlabel("시간 (ms)")
    ax.set_xlim(0, 31)
    ax.grid(axis="y", visible=False)


def main(out_dir: str | Path | None = None) -> list[Path]:
    ctx, span = load()
    st = load_structure()
    fig, ax_a = ps.new_figure("double", A_H + B_H, layout="none")
    H = A_H + B_H
    ax_a.set_position([0, B_H / H, 1, A_H / H])
    draw_structure(ax_a, st)
    fig.text(0.15 / FIG_W, 1 - 0.1 / H, "(a) YOLO11m 블록(model.0–23)과 HEF 컨텍스트", ha="left", va="top",
             fontsize=ps.TITLE_SIZE)
    bl, bb, br, bt = 2.0, 1.0, 1.1, 0.75     # panel (b) margins, cm
    ax_b = fig.add_axes([bl / FIG_W, bb / H, (FIG_W - bl - br) / FIG_W, (B_H - bb - bt) / H])
    draw_cost(ax_b, ctx)
    fig.text(0.15 / FIG_W, (B_H - 0.15) / H, "(b) 배치 구간의 컨텍스트별 고정비 a와 프레임당 비용 b", ha="left",
             va="top", fontsize=ps.TITLE_SIZE)
    paths = ps.save(fig, ps.resolve_out_dir(out_dir), STEM)

    sum_a = sum(ctx[c]["a"] for c in CONTEXTS)
    sum_b = sum(ctx[c]["b"] for c in CONTEXTS)
    lines = [f"(a) {c}: {_ctx_desc(st, c)}" for c in CONTEXTS]
    lines += [f"(a) split model.{b.index}: " + ", ".join(f"{c} {n} nodes" for c, n in b.shares)
              for b in st.blocks if len(b.shares) > 1]
    lines += [f"(a) head P{h.level + 3} (stride {h.stride}, from model.{h.src}): "
              + ", ".join(f"{c} {n} nodes" for c, n in h.shares) + "; outputs "
              + ", ".join(f"{o['branch']} {_shape(o['shape'])} {o['ctx']}" for o in h.outputs) for h in st.heads]
    lines += [f"(a) cross-context edge model.{s} -> model.{t}: {_shape(sh)}" for (s, t), sh in sorted(st.cross.items())]
    lines += [f"(b) {c}: a {ctx[c]['a']:.2f} ms ({100 * ctx[c]['a'] / span['a']:.1f}% of batch_span a), "
              f"b {ctx[c]['b']:.2f} ms, rms {ctx[c]['rms']:.3f}, n {ctx[c]['n']}" for c in CONTEXTS]
    lines.append(f"(b) sum of contexts: a {sum_a:.2f} ms, b {sum_b:.2f} ms")
    lines.append(f"(b) cross-check fits.totals.batch_span: a {span['a']:.2f}, b {span['b']:.2f}, rms {span['rms']:.2f} "
                 f"(difference in a = context transitions {span['a'] - sum_a:.3f} ms)")
    lines.append("not used: JSON fps, fits.totals.fps_period (do-not-cite)")
    ps.report(STEM, [LAYERS_SRC, SRC], lines)
    return paths


if __name__ == "__main__":
    sys.exit(ps.figure_cli(main, __doc__.splitlines()[0]))
