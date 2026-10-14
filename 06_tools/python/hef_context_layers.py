#!/usr/bin/env python3
"""Map the contexts of a multi-context HEF to the ONNX blocks (/model.N/...) they compute. Stdlib only.

Origin: written for 07_results/hef_context_profile (2026-10-14) as the evidence for figure 2 of the paper
(which YOLO11m blocks run in dynamic_0..3 of 03_models/yolov11m.hef). Reads the HEF file (and optionally a
measure-fw-actions JSON) only; it never opens the device.

Method (also written into both outputs):
  position-inferred  The complete per-context layer lists come from ProtoHEFHef.mapping (field 3). hef.proto
                     (HailoRT v4.24.0) declares that field as opaque `bytes` and libhailort never parses it, so
                     its inner layout was found by a schema-less protobuf walk and is NOT documented:
                       mapping.2.1[]      global layer records (every Hailo layer of the network group)
                       mapping.7.1[k]     context k: .1 header {1: "<ng>_context_<k>", 2: k, 3: ng name}
                                                     .2.1[] layer records of that context
                       layer record       1: layer type enum (meaning unknown), 2: Hailo layer name,
                                          45: repeated original (ONNX) node names
  schema cross-check hef.proto v4.24.0 fields: network_groups(2).ops(8).core_op(4).contexts(3).metadata(3)
                     .edge_layers(2) -> direction(1), layer_info(3) / layer_mux(4) with name(1) and
                     original_names(5 / 6), context_switch_info(5).edge_connection_type(3) / connected_contexts(8).
                     Edge layers are only the boundary, inter-context and DDR layers of a context (not all
                     layers), so they confirm the context split but cannot give it alone.
  fw-actions check   (--fw-actions) for each dynamic_k the counts of activate_{boundary,inter_context,
                     ddr_buffer}_{input,output} must equal the HEF edge layers of context k. hailortcli names
                     dynamic contexts by their index (download_action_list_command.cpp: "dynamic_{}").

Checks (exit code 1 when one fails; the outputs are still written):
  partition   the per-context lists are disjoint and their union equals the global layer list
  headers     mapping context k carries index k and name "<ng>_context_<k>"; context count = schema count
  edges       every original name of every schema edge layer of context k is an original name in mapping
              context k
  fw_actions  the activation counts above match (only with --fw-actions; its hef.file_hash must match too)
  coverage    every block 0..23 of the YOLO11 table appears in some context

Block types come from ultralytics yolo11.yaml (YOLO11_YAML_SOURCE). Scale m = [0.50, 1.00, 512]: repeats 2 -> 1,
and ultralytics tasks.py forces C3k2 c3k=True for scales m/l/x (C3k with m.0, m.1 inside). The HEF-side check
per block looks for characteristic ONNX node paths (BLOCK_SIGNATURES).

Outputs (fixed names in --out-dir; no timestamps, paths relative to the repository):
  context_layers.json   full data (per-context modules, ONNX nodes, Hailo layer counts, boundaries, edges, checks)
  context_layers.txt    human summary: one line per context and the YOLO11m block table

Usage:
  python3 06_tools/python/hef_context_layers.py [--hef 03_models/yolov11m.hef]
      [--fw-actions 07_results/hef_context_profile/raw/b1_run1/fwactions.json]
      [--out-dir build/out/hef_context_layers]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hef_bytes import CONN, DIRN, first, msg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DEFAULT_HEF = REPO / "03_models" / "yolov11m.hef"
DEFAULT_OUT = REPO / "build" / "out" / "hef_context_layers"
OUT_JSON = "context_layers.json"
OUT_TXT = "context_layers.txt"

HEF_MAGIC = 0x01484546
HEADER_SIZE = {0: 12 + 20, 1: 12 + 16, 2: 12 + 32, 3: 12 + 44}   # same table as hef_bytes.py

# ProtoHEFHwArch (hef.proto v4.24.0). yolov11m.hef has 1 (HAILO8P); hailortcli parse-hef prints it as HAILO8.
HW_ARCH = {0: "HAILO8", 1: "HAILO8P", 2: "HAILO8R", 3: "HAILO8L", 4: "HAILO15M", 5: "HAILO10H",
           6: "HAILO15L", 103: "HAILO15H"}

PROTO_SOURCE = "https://github.com/hailo-ai/hailort/blob/v4.24.0/hailort/libhailort/hef.proto"
YOLO11_YAML_SOURCE = ("https://github.com/ultralytics/ultralytics/blob/3131844a4a48f1266db52c4bb1f8adfc7d3e9d7f/"
                      "ultralytics/cfg/models/11/yolo11.yaml")
C3K_RULE_SOURCE = ("https://github.com/ultralytics/ultralytics/blob/9a1a1471ec0834041ca101d23302909809ed58bb/"
                   "ultralytics/nn/tasks.py parse_model(): 'if m is C3k2: if scale in {\"m\", \"l\", \"x\"}: c3k = True'")
METHOD = ("position-inferred: per-context layer lists from the opaque ProtoHEFHef.mapping bytes (field 3, not in "
          "hef.proto, layout found by position); cross-checked with schema edge layers (hef.proto v4.24.0 "
          "ContextMetadata.edge_layers original_names) and fw-actions activation counts")

# Schema field numbers (hef.proto v4.24.0).
HEF_NETWORK_GROUPS, HEF_MAPPING = 2, 3
NG_METADATA, NG_OPS = 1, 8
NGMETA_NAME = 1
OP_NAME, OP_INPUT_PADS, OP_CORE_OP, OP_NMS = 1, 2, 4, 5
COREOP_CONTEXTS = 3
CTX_INDEX, CTX_METADATA = 1, 3
CTXMETA_EDGE_LAYERS = 2
EDGE_DIRECTION, EDGE_INFO, EDGE_MUX, EDGE_PLANES, EDGE_CSI = 1, 3, 4, 8, 5
INFO_NAME, INFO_BASE, INFO_ORIG = 1, 2, 5
MUX_NAME, MUX_BASE, MUX_ORIG = 1, 2, 6
BASE_HEIGHT, BASE_WIDTH, BASE_FEATURES = 1, 3, 5
CSI_CONN, CSI_CONNECTED, CONNCTX_INDEX = 3, 8, 1
NMS_SCORE, NMS_IOU, NMS_MAX_PROPOSALS, NMS_CLASSES, NMS_YOLOV8 = 1, 2, 3, 4, 12
Y8_DECODERS, Y8_REG_LEN = 4, 5
Y8DEC_STRIDE, Y8DEC_REG_PAD, Y8DEC_CLS_PAD = 1, 2, 3

# Position-inferred field numbers inside ProtoHEFHef.mapping (not in hef.proto).
MAP_GLOBAL, MAP_CONTEXTS = 2, 7
MAPGLOBAL_LAYERS = 1
MAPCTXS_LIST = 1
MAPCTX_HEADER, MAPCTX_BODY = 1, 2
MAPHDR_NAME, MAPHDR_INDEX, MAPHDR_NG = 1, 2, 3
MAPBODY_LAYERS = 1
LAYER_TYPE, LAYER_NAME, LAYER_ORIG = 1, 2, 45

# fw-actions activation type per (connection type, direction) of a HEF edge layer.
FW_ACTION = {
    ("BOUNDARY", "H2D"): "activate_boundary_input",
    ("BOUNDARY", "D2H"): "activate_boundary_output",
    ("INTER_CONTEXT", "H2D"): "activate_inter_context_input",
    ("INTER_CONTEXT", "D2H"): "activate_inter_context_output",
    ("DDR", "H2D"): "activate_ddr_buffer_input",
    ("DDR", "D2H"): "activate_ddr_buffer_output",
}

# ultralytics yolo11.yaml: (index, from, repeats, module, args, comment).
YOLO11_BLOCKS = [
    (0, "-1", 1, "Conv", "[64, 3, 2]", "P1/2"),
    (1, "-1", 1, "Conv", "[128, 3, 2]", "P2/4"),
    (2, "-1", 2, "C3k2", "[256, False, 0.25]", ""),
    (3, "-1", 1, "Conv", "[256, 3, 2]", "P3/8"),
    (4, "-1", 2, "C3k2", "[512, False, 0.25]", ""),
    (5, "-1", 1, "Conv", "[512, 3, 2]", "P4/16"),
    (6, "-1", 2, "C3k2", "[512, True]", ""),
    (7, "-1", 1, "Conv", "[1024, 3, 2]", "P5/32"),
    (8, "-1", 2, "C3k2", "[1024, True]", ""),
    (9, "-1", 1, "SPPF", "[1024, 5]", ""),
    (10, "-1", 2, "C2PSA", "[1024]", ""),
    (11, "-1", 1, "nn.Upsample", '[None, 2, "nearest"]', ""),
    (12, "[-1, 6]", 1, "Concat", "[1]", "cat backbone P4"),
    (13, "-1", 2, "C3k2", "[512, False]", ""),
    (14, "-1", 1, "nn.Upsample", '[None, 2, "nearest"]', ""),
    (15, "[-1, 4]", 1, "Concat", "[1]", "cat backbone P3"),
    (16, "-1", 2, "C3k2", "[256, False]", "P3/8-small"),
    (17, "-1", 1, "Conv", "[256, 3, 2]", ""),
    (18, "[-1, 13]", 1, "Concat", "[1]", "cat head P4"),
    (19, "-1", 2, "C3k2", "[512, False]", "P4/16-medium"),
    (20, "-1", 1, "Conv", "[512, 3, 2]", ""),
    (21, "[-1, 10]", 1, "Concat", "[1]", "cat head P5"),
    (22, "-1", 2, "C3k2", "[1024, True]", "P5/32-large"),
    (23, "[16, 19, 22]", 1, "Detect", "[nc]", "Detect(P3, P4, P5)"),
]
SCALE_M_DEPTH = 0.50

# Substrings that a block's ONNX node paths (relative to /model.N/) must contain, per block type.
BLOCK_SIGNATURES = {
    "Conv": ["conv/Conv", "act/Sigmoid", "act/Mul"],
    "C3k2": ["cv1/conv/Conv", "Split", "m.0/", "Concat", "cv2/conv/Conv"],
    "SPPF": ["cv1/conv/Conv", "m/MaxPool", "m_1/MaxPool", "m_2/MaxPool", "Concat", "cv2/conv/Conv"],
    "C2PSA": ["cv1/conv/Conv", "Split", "attn/qkv/conv/Conv", "attn/MatMul", "attn/Softmax",
              "attn/proj/conv/Conv", "ffn.0/conv/Conv", "ffn.1/conv/Conv", "Concat", "cv2/conv/Conv"],
    "nn.Upsample": ["Resize"],
    "Concat": ["Concat"],
    "Detect": ["cv2.0/", "cv2.1/", "cv2.2/", "cv3.0/", "cv3.1/", "cv3.2/"],
}

MODULE_RE = re.compile(r"^/model\.(\d+)/(.+)$")
BOUNDARY_RE = re.compile(r"_context_(\d+)_to_.*_context_(\d+)_(\d+)$")


class HefError(ValueError):
    """The file is not a HEF this tool understands."""


def s(v) -> str:
    return bytes(v).decode("utf-8", "replace")


def repo_rel(p: Path) -> str:
    """Path relative to the repository when inside it, else as resolved (keeps outputs machine independent)."""
    rp = Path(p).resolve()
    try:
        return rp.relative_to(REPO).as_posix()
    except ValueError:
        return rp.as_posix()


def f64(v: int) -> float:
    """Reinterpret a fixed64 wire value as a double."""
    return struct.unpack("<d", struct.pack("<Q", v))[0]


# ---------------------------------------------------------------- HEF reading

def read_hef(raw: bytes) -> tuple[dict, dict]:
    """Return (header info, decoded top-level ProtoHEFHef message)."""
    if len(raw) < 12:
        raise HefError("file too short for a HEF header")
    magic, version, proto_size = struct.unpack_from(">III", raw, 0)
    if magic != HEF_MAGIC:
        raise HefError(f"bad magic 0x{magic:08x} (expected 0x{HEF_MAGIC:08x})")
    if version not in HEADER_SIZE:
        raise HefError(f"unsupported HEF header version {version}")
    hdr = HEADER_SIZE[version]
    if hdr + proto_size > len(raw):
        raise HefError(f"proto_size {proto_size} past end of file")
    hef = msg(memoryview(raw)[hdr:hdr + proto_size])
    header = dict(version=version, header_size=hdr, proto_offset=hdr, proto_size=proto_size)
    if hef.get(1):
        h = msg(hef[1][0])
        header["sdk_version"] = s(first(h, 3, b""))
        header["hw_arch_enum"] = first(h, 1)
        header["hw_arch"] = HW_ARCH.get(first(h, 1), "?")
    return header, hef


def edge_record(el) -> dict:
    """Decode one ProtoHEFEdgeLayer (schema) into a dict."""
    e = msg(el)
    csi = msg(first(e, EDGE_CSI, b""))
    if e.get(EDGE_INFO):
        info, kind, name_f, base_f, orig_f = msg(e[EDGE_INFO][0]), "info", INFO_NAME, INFO_BASE, INFO_ORIG
    elif e.get(EDGE_MUX):
        info, kind, name_f, base_f, orig_f = msg(e[EDGE_MUX][0]), "mux", MUX_NAME, MUX_BASE, MUX_ORIG
    else:
        info, kind = None, ("planes" if e.get(EDGE_PLANES) else "?")
    rec = dict(direction=DIRN.get(first(e, EDGE_DIRECTION), "?"),
               connection=CONN.get(first(csi, CSI_CONN), "?"), kind=kind, name="", original_names=[],
               shape=None, connected_contexts=[first(msg(c), CONNCTX_INDEX) for c in csi[CSI_CONNECTED]])
    if info is not None:
        rec["name"] = s(first(info, name_f, b""))
        rec["original_names"] = [s(x) for x in info[orig_f]]
        b = msg(first(info, base_f, b""))
        rec["shape"] = [first(b, BASE_HEIGHT), first(b, BASE_WIDTH), first(b, BASE_FEATURES)]
    return rec


def decode_schema(hef: dict) -> dict:
    """Network group name, schema contexts with edge layers, and host NMS ops (first network group)."""
    if not hef.get(HEF_NETWORK_GROUPS):
        raise HefError("no network_groups in the HEF")
    g = msg(hef[HEF_NETWORK_GROUPS][0])
    ng_name = s(first(msg(first(g, NG_METADATA, b"")), NGMETA_NAME, b""))
    contexts, nms_ops = [], []
    for op in g[NG_OPS]:
        om = msg(op)
        if om.get(OP_CORE_OP):
            co = msg(om[OP_CORE_OP][0])
            for pos, c in enumerate(co[COREOP_CONTEXTS]):
                cm = msg(c)
                meta = msg(first(cm, CTX_METADATA, b""))
                contexts.append(dict(position=pos, context_index=first(cm, CTX_INDEX),
                                     edges=[edge_record(el) for el in meta[CTXMETA_EDGE_LAYERS]]))
        elif om.get(OP_NMS):
            n = msg(om[OP_NMS][0])
            nms = dict(name=s(first(om, OP_NAME, b"")), score_threshold=f64(first(n, NMS_SCORE)),
                       iou_threshold=f64(first(n, NMS_IOU)), max_proposals_per_class=first(n, NMS_MAX_PROPOSALS),
                       classes=first(n, NMS_CLASSES), input_pads=len(om[OP_INPUT_PADS]), type="?")
            if n.get(NMS_YOLOV8):
                y = msg(n[NMS_YOLOV8][0])
                nms.update(type="yolov8", regression_length=first(y, Y8_REG_LEN),
                           bbox_decoders=[dict(stride=first(msg(d), Y8DEC_STRIDE),
                                               reg_pad_index=first(msg(d), Y8DEC_REG_PAD),
                                               cls_pad_index=first(msg(d), Y8DEC_CLS_PAD))
                                          for d in y[Y8_DECODERS]])
            nms_ops.append(nms)
    if not contexts:
        raise HefError("no core_op contexts found (single-context or legacy HEF layout is not supported)")
    return dict(network_group=ng_name, contexts=contexts, nms_ops=nms_ops)


def layer_record(b) -> dict:
    m = msg(b)
    return dict(name=s(first(m, LAYER_NAME, b"")), type=first(m, LAYER_TYPE),
                original_names=[s(x) for x in m[LAYER_ORIG]])


def decode_mapping(hef: dict) -> dict:
    """Global and per-context layer records from the opaque mapping bytes (position-inferred layout)."""
    if not hef.get(HEF_MAPPING):
        raise HefError("ProtoHEFHef.mapping (field 3) is absent: no per-context layer lists in this HEF")
    mp = msg(hef[HEF_MAPPING][0])
    if not mp.get(MAP_GLOBAL) or not mp.get(MAP_CONTEXTS):
        raise HefError("mapping has no fields 2/7: unknown mapping layout (position inference does not hold)")
    glob = [layer_record(b) for b in msg(mp[MAP_GLOBAL][0])[MAPGLOBAL_LAYERS]]
    ctxs = []
    for entry in mp[MAP_CONTEXTS]:
        for c in msg(entry)[MAPCTXS_LIST]:
            cm = msg(c)
            h = msg(first(cm, MAPCTX_HEADER, b""))
            body = msg(first(cm, MAPCTX_BODY, b""))
            ctxs.append(dict(name=s(first(h, MAPHDR_NAME, b"")), index=first(h, MAPHDR_INDEX),
                             network_group=s(first(h, MAPHDR_NG, b"")),
                             layers=[layer_record(b) for b in body[MAPBODY_LAYERS]]))
    return dict(global_layers=glob, contexts=ctxs)


# ---------------------------------------------------------------- ONNX path helpers

def module_of(onnx_name: str) -> tuple[int, str] | None:
    """('/model.6/m.0/cv1/conv/Conv') -> (6, 'm.0/cv1/conv/Conv'); None for names outside /model.N/."""
    m = MODULE_RE.match(onnx_name)
    return (int(m.group(1)), m.group(2)) if m else None


def collapse(rel_paths: list[str], all_rel_paths: list[str]) -> list[str]:
    """Shortest path prefixes that cover rel_paths exactly, given all paths of the block (first-seen order).

    A prefix is emitted when every node of the block under it is in rel_paths; otherwise its children are
    examined. 'conv'/'act' leaf groups of a Conv module are therefore reported as their parent (e.g. 'cv1').
    """
    present = set(rel_paths)
    split_all = [p.split("/") for p in all_rel_paths]
    out: list[str] = []

    def walk(prefix: tuple[str, ...]) -> None:
        under = [p for p in split_all if tuple(p[:len(prefix)]) == prefix]
        mine = ["/".join(p) for p in under if "/".join(p) in present]
        if not mine:
            return
        if len(mine) == len(under) and prefix:
            out.append("/".join(prefix))
            return
        children: list[str] = []
        for p in under:
            if len(p) > len(prefix) and p[len(prefix)] not in children:
                children.append(p[len(prefix)])
        for ch in children:
            walk(prefix + (ch,))

    walk(())
    return out


def ranges(nums: list[int]) -> str:
    """[0,1,2,3,5] -> 'model.0-3, model.5'."""
    parts, i = [], 0
    nums = sorted(nums)
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"model.{nums[i]}" + (f"-{nums[j]}" if j > i else ""))
        i = j + 1
    return ", ".join(parts) if parts else "-"


def match_layer(edge_name: str, layer_names: list[str]) -> str | None:
    """Mapping layer that a schema edge layer name refers to: exact, split sub-layer (name_*), or suffixed edge."""
    if edge_name in layer_names:
        return edge_name
    subs = [n for n in layer_names if n.startswith(edge_name + "_")]
    if subs:
        return subs[0]
    bases = [n for n in layer_names if edge_name.startswith(n + "_")]
    return max(bases, key=len) if bases else None


# ---------------------------------------------------------------- report

def block_check(btype: str, rels: list[str]) -> dict:
    """Signature check of one block's ONNX node paths against its yolo11.yaml type."""
    sig = BLOCK_SIGNATURES.get(btype, [])
    missing = [t for t in sig if not any(t in r for r in rels)]
    ops = []
    for r in rels:
        op = r.split("/")[-1]
        if op not in ops:
            ops.append(op)
    notes = []
    if btype == "C3k2":
        top = sorted({r.split("/")[0] for r in rels if re.match(r"^m\.\d+/", r)})
        notes.append(f"top-level repeats {len(top)} ({', '.join(top)})")
        if any("/m/m.0/" in "/" + r for r in rels):
            inner = sorted({r.split("/")[2] for r in rels if re.match(r"^m\.\d+/m/m\.\d+/", r)})
            notes.append(f"C3k inside (m.0/m/{{{','.join(inner)}}}): c3k=True")
        else:
            notes.append("plain Bottleneck inside: c3k=False")
    elif btype == "C2PSA":
        top = sorted({r.split("/")[1] for r in rels if re.match(r"^m/m\.\d+/", r)})
        notes.append(f"PSABlock repeats {len(top)} ({', '.join(top)})")
        if not any("attn/pe" in r for r in rels):
            notes.append("no 'attn/pe' node name in the HEF")
    elif btype == "Detect":
        if not any(("dfl" in r.lower()) for r in rels):
            notes.append("no DFL/decode nodes on chip: only cv2.i/cv3.i convs (decode+NMS = host nms_op)")
    return dict(signature_missing=missing, onnx_ops=ops, notes=notes)


def fw_counts(fw: dict) -> dict:
    """Activation action counts per dynamic_k of the first run."""
    runs = fw.get("runs") or []
    if not runs:
        raise HefError("fw-actions JSON has no runs")
    out = {}
    for c in runs[0].get("contexts", []):
        name = c.get("context_name", "")
        if re.match(r"^dynamic_\d+$", name) and name not in out:
            cnt = {t: 0 for t in FW_ACTION.values()}
            for a in c.get("actions", []):
                if a.get("type") in cnt:
                    cnt[a["type"]] += 1
            out[name] = cnt
    return out


def build_report(raw: bytes, hef_label: str, fw: dict | None = None, fw_label: str | None = None) -> dict:
    header, hef = read_hef(raw)
    schema = decode_schema(hef)
    mapping = decode_mapping(hef)
    ng = schema["network_group"]

    # Block-wide ONNX node order (first appearance over contexts in order) and the set per block.
    all_rel: dict[int, list[str]] = {}
    for c in mapping["contexts"]:
        for lay in c["layers"]:
            for on in lay["original_names"]:
                mo = module_of(on)
                if mo and mo[1] not in all_rel.setdefault(mo[0], []):
                    all_rel[mo[0]].append(mo[1])

    fwc = fw_counts(fw) if fw is not None else None
    contexts, checks_detail = [], {"edges": [], "headers": []}
    seen_names: list[str] = []
    for k, mc in enumerate(mapping["contexts"]):
        names = [lay["name"] for lay in mc["layers"]]
        seen_names += names
        orig_set = {on for lay in mc["layers"] for on in lay["original_names"]}
        rel_by_mod: dict[int, list[str]] = {}
        nodes_by_mod: dict[int, list[str]] = {}
        other_orig: list[str] = []
        for lay in mc["layers"]:
            for on in lay["original_names"]:
                mo = module_of(on)
                if mo is None:
                    if on not in other_orig:
                        other_orig.append(on)
                    continue
                if mo[1] not in rel_by_mod.setdefault(mo[0], []):
                    rel_by_mod[mo[0]].append(mo[1])
                    nodes_by_mod.setdefault(mo[0], []).append(on)
        mods = sorted(rel_by_mod)
        full = [n for n in mods if set(rel_by_mod[n]) == set(all_rel[n])]
        partial = [n for n in mods if n not in full]
        exp_name = f"{ng}_context_{k}"
        hdr_ok = mc["index"] == k and mc["name"] == exp_name and mc["network_group"] == ng
        if not hdr_ok:
            checks_detail["headers"].append(f"mapping context #{k}: index {mc['index']} name {mc['name']!r}")

        boundaries = []
        for n in names:
            bm = BOUNDARY_RE.search(n)
            if bm:
                src, dst = int(bm.group(1)), int(bm.group(2))
                boundaries.append(dict(name=n, src=src, dst=dst, id=int(bm.group(3)),
                                       role="send" if src == k else ("recv" if dst == k else "?")))
        ddr_layers = [n for n in names if "ddr" in n.lower()]

        sc = schema["contexts"][k] if k < len(schema["contexts"]) else None
        edges, edge_mods = [], set()
        if sc is not None:
            for e in sc["edges"]:
                short = e["name"].split("/", 1)[1] if e["name"].startswith(ng + "/") else e["name"]
                bad = [on for on in e["original_names"] if on not in orig_set]
                if bad:
                    checks_detail["edges"].append(f"context {k} edge {e['name']}: not in mapping context: {bad}")
                for on in e["original_names"]:
                    mo = module_of(on)
                    if mo:
                        edge_mods.add(mo[0])
                rec = dict(e)
                rec["mapping_layer"] = match_layer(short, names)
                rec["original_names_in_context"] = not bad
                if e["direction"] == "D2H" and e["connection"] == "BOUNDARY" and e["shape"] and e["shape"][0]:
                    in_h = next((x["shape"][0] for c2 in schema["contexts"] for x in c2["edges"]
                                 if x["connection"] == "BOUNDARY" and x["direction"] == "H2D" and x["shape"]), 0)
                    rec["stride"] = in_h // e["shape"][0] if in_h else None
                edges.append(rec)
        hef_counts = {t: 0 for t in FW_ACTION.values()}
        for e in edges:
            t = FW_ACTION.get((e["connection"], e["direction"]))
            if t:
                hef_counts[t] += 1
        fw_name = f"dynamic_{k}"
        fw_entry = None
        if fwc is not None:
            got = fwc.get(fw_name)
            fw_entry = dict(context=fw_name, fw_counts=got, hef_edge_counts=hef_counts,
                            match=got == hef_counts)

        contexts.append(dict(
            index=k, mapping_name=mc["name"], fw_context=fw_name,
            schema_context_index=sc["context_index"] if sc else None,
            modules_full=full, modules_partial=partial,
            partial_paths={str(n): collapse(rel_by_mod[n], all_rel[n]) for n in partial},
            module_paths={str(n): rel_by_mod[n] for n in mods},
            onnx_nodes={str(n): nodes_by_mod[n] for n in mods},
            other_original_names=other_orig,
            hailo_layers=dict(total=len(names), with_original_names=sum(1 for l in mc["layers"]
                                                                        if l["original_names"]),
                              generated=sum(1 for l in mc["layers"] if not l["original_names"])),
            boundaries=boundaries, ddr_layers=ddr_layers,
            schema_edges=edges, schema_edge_modules=sorted(edge_mods),
            fw_actions=fw_entry))

    glob_names = [lay["name"] for lay in mapping["global_layers"]]
    dup = sorted({n for n in seen_names if seen_names.count(n) > 1})
    partition_ok = not dup and sorted(seen_names) == sorted(glob_names)
    headers_ok = not checks_detail["headers"] and len(mapping["contexts"]) == len(schema["contexts"])
    edges_ok = not checks_detail["edges"]
    present = {n for c in contexts for n in c["module_paths"]}
    missing = [b[0] for b in YOLO11_BLOCKS if str(b[0]) not in present]
    span = {str(n): [c["index"] for c in contexts if str(n) in c["module_paths"]] for n in sorted(all_rel)}

    fw_info, fw_ok = None, None
    if fw is not None:
        fw_hash = (fw.get("hef") or {}).get("file_hash")
        hef_md5 = hashlib.md5(raw).hexdigest()
        fw_ok = fw_hash == hef_md5 and all(c["fw_actions"]["match"] for c in contexts)
        fw_info = dict(path=fw_label, hef_file_hash=fw_hash, hef_hash_match=fw_hash == hef_md5,
                       run_used=0, batch_size=(fw.get("runs") or [{}])[0].get("batch_size"))

    blocks = []
    for idx, frm, rep, mod, args, comment in YOLO11_BLOCKS:
        rels = all_rel.get(idx, [])
        chk = block_check(mod, rels)
        eff = rep if rep == 1 else max(round(rep * SCALE_M_DEPTH), 1)
        if mod == "C2PSA" and "no 'attn/pe' node name in the HEF" in chk["notes"]:
            # Unnamed DDR edges in the block's context(s) are the only unnamed on-chip buffers (inferred link).
            cand = [f"{e['name'].split('/')[-1]} {'x'.join(map(str, e['shape']))} (context {c['index']})"
                    for c in contexts if str(idx) in c["module_paths"] for e in c["schema_edges"]
                    if e["connection"] == "DDR" and e["direction"] == "D2H" and not e["original_names"]]
            if cand:
                chk["notes"].append(f"unnamed DDR buffer in the same context: {', '.join(cand)} "
                                    "- candidate for attn/pe (3x3 depthwise unrolled to 9*C features; inferred)")
        blocks.append(dict(index=idx, module=mod, yaml_from=frm, yaml_repeats=rep, scale_m_repeats=eff,
                           yaml_args=args, yaml_comment=comment, contexts=span.get(str(idx), []),
                           onnx_nodes_in_hef=len(rels), **chk))

    return dict(
        tool="06_tools/python/hef_context_layers.py",
        method=METHOD,
        sources=dict(hef_proto=PROTO_SOURCE, yolo11_yaml=YOLO11_YAML_SOURCE, c3k_rule=C3K_RULE_SOURCE),
        hef=dict(path=hef_label, md5=hashlib.md5(raw).hexdigest(), size=len(raw), **header,
                 network_group=ng, contexts=len(schema["contexts"])),
        fw_actions=fw_info,
        position_inferred_fields=dict(
            mapping="ProtoHEFHef field 3 (bytes in hef.proto; not parsed by libhailort 4.24)",
            global_layers="mapping.2.1[] layer records",
            contexts="mapping.7.1[k]: .1 header {1 name, 2 index, 3 network group}, .2.1[] layer records",
            layer_record="1 type enum (meaning unknown), 2 Hailo layer name, 45 repeated original (ONNX) names"),
        global_layers=len(glob_names),
        contexts=contexts,
        module_contexts=span,
        nms_ops=schema["nms_ops"],
        blocks=blocks,
        checks=dict(partition=partition_ok, headers=headers_ok, edges=edges_ok, fw_actions=fw_ok,
                    coverage=not missing,
                    detail=dict(duplicates=dup, headers=checks_detail["headers"], edges=checks_detail["edges"],
                                missing_blocks=missing)),
    )


def render_text(r: dict) -> str:
    h = r["hef"]
    L = [f"# HEF context -> YOLO11m block map ({r['tool']})",
         f"hef: {h['path']}  md5 {h['md5']}  size {h['size']}  header v{h['version']}  "
         f"proto @{h['proto_offset']} size {h['proto_size']}  sdk {h.get('sdk_version', '?')}  "
         f"hw_arch {h.get('hw_arch', '?')}",
         f"network group: {h['network_group']}, contexts: {h['contexts']}, Hailo layers (global list): "
         f"{r['global_layers']}",
         f"method: {r['method']}",
         f"fw-actions: {r['fw_actions']['path'] if r['fw_actions'] else '(not given)'}",
         "",
         "## Contexts (full = every ONNX node of the block named in the HEF is in this context)"]
    for c in r["contexts"]:
        part = "; ".join(f"model.{n} [{', '.join(c['partial_paths'][str(n)])}]" for n in c["modules_partial"])
        L.append(f"context {c['index']} ({c['fw_context']}): {ranges(c['modules_full'])}"
                 + (f" | partial: {part}" if part else ""))
    L += ["", "## Per-context detail"]
    for c in r["contexts"]:
        hl = c["hailo_layers"]
        L.append(f"context {c['index']} ({c['mapping_name']}): Hailo layers {hl['total']} "
                 f"(with ONNX names {hl['with_original_names']}, compiler-generated {hl['generated']})")
        if c["other_original_names"]:
            L.append(f"  non-/model original names: {', '.join(c['other_original_names'])}")
        sends = [f"{b['src']}->{b['dst']} #{b['id']}" for b in c["boundaries"] if b["role"] == "send"]
        recvs = [f"{b['src']}->{b['dst']} #{b['id']}" for b in c["boundaries"] if b["role"] == "recv"]
        L.append(f"  context boundary layers: send [{', '.join(sends) or '-'}]  recv [{', '.join(recvs) or '-'}]")
        if c["ddr_layers"]:
            L.append(f"  DDR layers: {', '.join(c['ddr_layers'])}")
        for e in c["schema_edges"]:
            shp = "x".join(str(x) for x in e["shape"]) if e["shape"] else "?"
            mods = sorted({module_of(o)[0] for o in e["original_names"] if module_of(o)})
            extra = f" stride {e['stride']}" if e.get("stride") else ""
            to = f" ctx{e['connected_contexts']}" if e["connected_contexts"] else ""
            L.append(f"  edge {e['direction']} {e['connection']:<13} {e['name']} {shp}{to}{extra} "
                     f"orig-> {', '.join(f'model.{m}' for m in mods) or '-'}"
                     f"{'' if e['original_names_in_context'] else '  !! not in mapping context'}")
        fa = c["fw_actions"]
        if fa:
            short = {k.replace("activate_", ""): v for k, v in fa["hef_edge_counts"].items() if v}
            L.append(f"  fw-actions {fa['context']}: {'match' if fa['match'] else 'MISMATCH'} "
                     f"(hef edges {short}; fw {({k.replace('activate_', ''): v for k, v in (fa['fw_counts'] or {}).items() if v})})")
    for n in r["nms_ops"]:
        dec = ", ".join(f"stride {d['stride']}" for d in n.get("bbox_decoders", []))
        L.append(f"host op {n['name']}: type {n['type']}, score {n['score_threshold']:g}, iou "
                 f"{n['iou_threshold']:g}, classes {n['classes']}, max/class {n['max_proposals_per_class']}, "
                 f"regression_length {n.get('regression_length', '?')}, decoders [{dec}], input pads {n['input_pads']}")
    L += ["", "## YOLO11m blocks",
          f"source: {r['sources']['yolo11_yaml']} (scale m = [0.50, 1.00, 512]; repeats 2 -> 1)",
          f"c3k rule: {r['sources']['c3k_rule']}",
          f"{'idx':>3}  {'module':<12} {'from':<13} {'args':<22} {'ctx':<6} {'nodes':>5}  HEF check"]
    for b in r["blocks"]:
        chk = "ok" if not b["signature_missing"] else f"missing {b['signature_missing']}"
        notes = ("; " + "; ".join(b["notes"])) if b["notes"] else ""
        ctx = ",".join(str(x) for x in b["contexts"]) or "-"
        tag = f"  # {b['yaml_comment']}" if b["yaml_comment"] else ""
        L.append(f"{b['index']:>3}  {b['module']:<12} {b['yaml_from']:<13} {b['yaml_args']:<22} {ctx:<6} "
                 f"{b['onnx_nodes_in_hef']:>5}  {chk}{notes}{tag}")
    ck = r["checks"]
    L += ["", "## Checks"]
    for k in ("partition", "headers", "edges", "fw_actions", "coverage"):
        v = ck[k]
        L.append(f"{k}: {'skipped' if v is None else ('ok' if v else 'FAIL')}")
    d = ck["detail"]
    for k in ("duplicates", "headers", "edges", "missing_blocks"):
        if d[k]:
            L.append(f"  {k}: {d[k]}")
    L.append("note: the per-context lists rely on the position-inferred mapping layout; the schema edge layers "
             "and fw-actions counts above are the independent cross-checks.")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Map HEF contexts to ONNX blocks (/model.N/...) from the HEF file only (no device). "
                    f"Writes {OUT_JSON} and {OUT_TXT} into --out-dir.")
    ap.add_argument("--hef", default=str(DEFAULT_HEF), help="HEF file (default 03_models/yolov11m.hef)")
    ap.add_argument("--fw-actions", help="measure-fw-actions JSON for the activation-count cross-check (optional)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT),
                    help="output directory (default build/out/hef_context_layers)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    inputs = {Path(args.hef).resolve()} | ({Path(args.fw_actions).resolve()} if args.fw_actions else set())
    if {(out_dir / OUT_JSON).resolve(), (out_dir / OUT_TXT).resolve()} & inputs:
        ap.error("refusing to overwrite an input file")
    try:
        raw = Path(args.hef).read_bytes()
        fw = json.loads(Path(args.fw_actions).read_text()) if args.fw_actions else None
        report = build_report(raw, repo_rel(Path(args.hef)), fw,
                              repo_rel(Path(args.fw_actions)) if args.fw_actions else None)
    except (OSError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / OUT_JSON).write_text(json.dumps(report, indent=1, ensure_ascii=True) + "\n")
    text = render_text(report)
    (out_dir / OUT_TXT).write_text(text)
    sys.stdout.write(text)
    ck = report["checks"]
    return 1 if any(ck[k] is False for k in ("partition", "headers", "edges", "fw_actions", "coverage")) else 0


if __name__ == "__main__":
    sys.exit(main())
