"""Tests for 06_tools/python/hef_context_layers.py (stdlib unittest, no pytest).

Run: python3 -m unittest discover -s 06_tools/python/tests

A tiny protobuf encoder builds a synthetic two-context HEF (header v2 + ProtoHEFHef with network_groups and the
position-inferred mapping layout) so the decoder, the full/partial split, the checks and the fw-actions
comparison are tested without the real model. One optional test reads 03_models/yolov11m.hef (git-ignored)
and is skipped when it is absent.
"""
from __future__ import annotations

import contextlib
import io
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hef_context_layers as hcl  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REAL_HEF = REPO / "03_models" / "yolov11m.hef"
REAL_FW = REPO / "07_results" / "hef_context_profile" / "raw" / "b1_run1" / "fwactions.json"


# ---------------------------------------------------------------- protobuf encoder

def varint(n: int) -> bytes:
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def f(no: int, val) -> bytes:
    """One field: int -> varint (wire 0), str/bytes -> length-delimited (wire 2)."""
    if isinstance(val, int):
        return varint(no << 3) + varint(val)
    data = val.encode() if isinstance(val, str) else val
    return varint((no << 3) | 2) + varint(len(data)) + data


def rep(no: int, vals) -> bytes:
    return b"".join(f(no, v) for v in vals)


def layer(name: str, origs: list[str], typ: int = 0) -> bytes:
    return f(1, typ) + f(2, name) + rep(45, origs)


def edge(direction: int, conn: int, name: str, origs: list[str], shape=(4, 4, 8), connected=()) -> bytes:
    base = f(1, shape[0]) + f(3, shape[1]) + f(5, shape[2]) + f(12, 1)
    info = f(1, name) + f(2, base) + rep(5, origs)
    csi = f(3, conn) + rep(8, [f(1, c) for c in connected])
    return f(1, direction) + f(3, info) + f(5, csi)


CTX_LAYERS = [
    [layer("input_layer1", ["images"]),
     layer("conv1", ["/model.0/conv/Conv", "/model.0/act/Mul"]),
     layer("conv2", ["/model.1/cv1/conv/Conv"]),
     layer("net_net_context_0_to_net_context_1_10", [], 26)],
    [layer("net_net_context_0_to_net_context_1_11", [], 25),
     layer("conv3_d0", ["/model.1/cv2/conv/Conv"]),
     layer("conv4", ["/model.2/Concat"])],
]
CTX_EDGES = [
    [edge(0, 0, "net/input_layer1", ["images"], (64, 64, 3)),
     edge(1, 1, "net/conv2", ["/model.1/cv1/conv/Conv"], connected=(1,))],
    [edge(0, 1, "net/net_net_context_0_to_net_context_1_11", ["/model.1/cv2/conv/Conv"], connected=(0,)),
     edge(1, 0, "net/conv4_9", ["/model.2/Concat"], (8, 8, 4))],
]


def make_hef(ctx_header_index_shift: int = 0) -> bytes:
    contexts = [f(1, k) + f(3, rep(2, CTX_EDGES[k])) for k in range(2)]
    core_op = rep(3, contexts)
    ng = f(1, f(1, "net")) + f(8, f(1, "") + f(4, core_op))
    glob = rep(1, [l for ls in CTX_LAYERS for l in ls])
    map_ctxs = b"".join(
        f(1, f(1, f(1, f"net_context_{k}") + f(2, k + ctx_header_index_shift) + f(3, "net"))
          + f(2, rep(1, CTX_LAYERS[k])))
        for k in range(2))
    mapping = f(2, glob) + f(7, map_ctxs)
    proto = f(1, f(3, "3.34.0")) + f(2, ng) + f(3, mapping)
    return struct.pack(">III", hcl.HEF_MAGIC, 2, len(proto)) + bytes(32) + proto


def make_fw(boundary_out_1: int = 1, file_hash: str | None = None) -> dict:
    def ctx(name, types):
        return {"context_name": name, "actions": [{"type": t} for t in types]}
    return {"hef": {"file_hash": file_hash},
            "runs": [{"batch_size": 1, "contexts": [
                ctx("activation", ["activate_boundary_input"]),
                ctx("dynamic_0", ["activate_boundary_input", "activate_inter_context_output", "lcu_interrupt"]),
                ctx("dynamic_1", ["activate_inter_context_input"] + ["activate_boundary_output"] * boundary_out_1),
            ]}]}


class SyntheticHef(unittest.TestCase):
    def setUp(self):
        self.raw = make_hef()
        import hashlib
        self.md5 = hashlib.md5(self.raw).hexdigest()

    def test_split_full_partial(self):
        r = hcl.build_report(self.raw, "synthetic.hef")
        c0, c1 = r["contexts"]
        self.assertEqual(c0["modules_full"], [0])
        self.assertEqual(c0["modules_partial"], [1])
        self.assertEqual(c0["partial_paths"]["1"], ["cv1"])
        self.assertEqual(c1["modules_full"], [2])
        self.assertEqual(c1["partial_paths"]["1"], ["cv2"])
        self.assertEqual(c0["hailo_layers"], {"total": 4, "with_original_names": 3, "generated": 1})
        self.assertEqual(c0["other_original_names"], ["images"])
        self.assertEqual([(b["src"], b["dst"], b["role"]) for b in c0["boundaries"]], [(0, 1, "send")])
        self.assertEqual([(b["src"], b["dst"], b["role"]) for b in c1["boundaries"]], [(0, 1, "recv")])
        self.assertEqual(c1["schema_edges"][1]["mapping_layer"], "conv4")      # suffixed edge name
        self.assertEqual(c1["schema_edges"][1]["stride"], 8)                   # 64 / 8
        ck = r["checks"]
        self.assertTrue(ck["partition"] and ck["headers"] and ck["edges"])
        self.assertIsNone(ck["fw_actions"])
        self.assertFalse(ck["coverage"])                                       # only blocks 0..2 present
        self.assertEqual(ck["detail"]["missing_blocks"], list(range(3, 24)))

    def test_fw_actions_match_and_mismatch(self):
        ok = hcl.build_report(self.raw, "s.hef", make_fw(1, self.md5), "fw.json")
        self.assertTrue(ok["checks"]["fw_actions"])
        bad = hcl.build_report(self.raw, "s.hef", make_fw(2, self.md5), "fw.json")
        self.assertFalse(bad["checks"]["fw_actions"])
        self.assertFalse(bad["contexts"][1]["fw_actions"]["match"])
        wrong_hash = hcl.build_report(self.raw, "s.hef", make_fw(1, "0" * 32), "fw.json")
        self.assertFalse(wrong_hash["checks"]["fw_actions"])

    def test_header_check_fails_on_wrong_index(self):
        r = hcl.build_report(make_hef(ctx_header_index_shift=1), "s.hef")
        self.assertFalse(r["checks"]["headers"])

    def test_bad_magic(self):
        with self.assertRaises(hcl.HefError):
            hcl.build_report(b"\x00" * 64, "x.hef")

    def test_collapse(self):
        all_p = ["cv1/conv/Conv", "cv1/act/Mul", "Split", "m.0/cv1/conv/Conv", "m.0/m/m.0/cv1/conv/Conv",
                 "m.0/m/m.0/Add", "m.0/m/m.1/cv1/conv/Conv"]
        got = hcl.collapse(["cv1/conv/Conv", "cv1/act/Mul", "Split", "m.0/m/m.0/cv1/conv/Conv"], all_p)
        self.assertEqual(got, ["cv1", "Split", "m.0/m/m.0/cv1"])
        self.assertEqual(hcl.ranges([0, 1, 2, 5, 7, 8]), "model.0-2, model.5, model.7-8")

    def test_main_writes_byte_identical_outputs(self):
        with tempfile.TemporaryDirectory() as td:
            hef = Path(td) / "s.hef"
            hef.write_bytes(self.raw)
            fw = Path(td) / "fw.json"
            fw.write_text(json.dumps(make_fw(1, self.md5)))
            outs = []
            for n in (1, 2):
                od = Path(td) / f"out{n}"
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = hcl.main(["--hef", str(hef), "--fw-actions", str(fw), "--out-dir", str(od)])
                self.assertEqual(rc, 1)                                        # coverage fails on purpose
                outs.append([(od / hcl.OUT_JSON).read_bytes(), (od / hcl.OUT_TXT).read_bytes()])
            self.assertEqual(outs[0], outs[1])
            self.assertEqual(sorted(p.name for p in (Path(td) / "out1").iterdir()),
                             sorted([hcl.OUT_JSON, hcl.OUT_TXT]))


@unittest.skipUnless(REAL_HEF.exists() and REAL_FW.exists(), "03_models/yolov11m.hef or fw-actions JSON absent")
class RealHef(unittest.TestCase):
    def test_yolov11m_mapping(self):
        r = hcl.build_report(REAL_HEF.read_bytes(), "03_models/yolov11m.hef",
                             json.loads(REAL_FW.read_text()), "fw.json")
        got = [(c["modules_full"], c["modules_partial"]) for c in r["contexts"]]
        self.assertEqual(got, [([0, 1, 2, 3, 4, 5], [6]), ([7, 8, 9, 10], [6]),
                               (list(range(11, 22)), [23]), ([22], [23])])
        self.assertEqual([c["hailo_layers"]["total"] for c in r["contexts"]], [190, 197, 223, 70])
        ck = r["checks"]
        self.assertTrue(all(ck[k] for k in ("partition", "headers", "edges", "fw_actions", "coverage")))


if __name__ == "__main__":
    unittest.main()
