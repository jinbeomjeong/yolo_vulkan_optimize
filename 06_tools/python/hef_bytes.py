#!/usr/bin/env python3
"""Minimal HEF (HailoRT hef.proto, v4.24.0) decoder: per-context config (CCW) bytes and edge-layer
per-frame transfer sizes. Stdlib only. Field numbers from hailort/libhailort/hef.proto @ v4.24.0.

Origin: written for the 07_results/candidate_reduction_survey experiment (2026-10-08) to answer
"how many bytes cross PCIe per frame / per batch" for 03_models/yolov11m.hef. Its stdout was saved as
07_results/candidate_reduction_survey/evidence/hef_proto_bytes.txt (a 5-line comment header and the
source listings were added by hand around it). hef_byte_model.py imports this module.

Input:  a .hef file (header + protobuf + CCW section; header versions 0..3).
Output: stdout only. Per network group: core-op variants, preliminary config bytes, and for each dynamic
        context the config bytes per cfg channel and every edge layer (direction, connection type, shape,
        frame / padded / core_bytes_per_buffer*core_buffers_per_frame sizes), then totals.
        The HEF path is echoed as given (the archived output was made with an absolute path).

Usage:
  python3 06_tools/python/hef_bytes.py 03_models/yolov11m.hef > build/out/hef_tools/hef_proto_bytes.txt
"""
import argparse
import collections
import struct
import sys


def varint(b, i):
    r = s = 0
    while True:
        x = b[i]; i += 1
        r |= (x & 0x7f) << s; s += 7
        if x < 0x80: return r, i

def fields(b):
    """yield (field_no, wire_type, value) ; value = int for varint/fixed, memoryview for len-delimited"""
    i, n = 0, len(b)
    while i < n:
        k, i = varint(b, i); f, w = k >> 3, k & 7
        if w == 0: v, i = varint(b, i)
        elif w == 1: v = struct.unpack_from('<Q', b, i)[0]; i += 8
        elif w == 5: v = struct.unpack_from('<I', b, i)[0]; i += 4
        elif w == 2:
            L, i = varint(b, i); v = b[i:i+L]; i += L
        else: raise ValueError(f'wire type {w}')
        yield f, w, v

def msg(b) -> collections.defaultdict:
    """Decode one protobuf message into {field_no: [values...]}."""
    d = collections.defaultdict(list)
    for f, w, v in fields(b): d[f].append(v)
    return d

def first(d, f, default=0):
    """First value of field f in a decoded message, or default when absent."""
    return d[f][0] if d.get(f) else default

CONN = {0: 'BOUNDARY', 1: 'INTER_CONTEXT', 2: 'DDR', 3: 'CACHE'}
DIRN = {0: 'H2D', 1: 'D2H'}

def base_info(b) -> dict:
    """Decode ProtoHEFEdgeLayerBase into a dict of shape / buffer fields."""
    m = msg(b)
    keys = dict(height=1, padded_height=2, width=3, padded_width=4, features=5, padded_features=6, format=7,
                sys_index=8, core_bytes_per_buffer=9, core_buffers_per_frame=10, data_bytes=12, max_shmifo_size=15)
    return {k: first(m, f) for k, f in keys.items()}

def align8(x): return (x + 7) // 8 * 8

def edge_layers(ctx_meta) -> list[dict]:
    """List the edge layers of one context's metadata (direction, connection type, name, base info)."""
    out = []
    for el in msg(ctx_meta)[2]:
        e = msg(el)
        direction = DIRN[first(e, 1)]
        csi = msg(first(e, 5, b''))
        conn = CONN.get(first(csi, 3), '?')
        connected = [first(msg(c), 1) for c in csi[8]]
        info = None; kind = 'info'
        if e.get(3): info = msg(e[3][0])
        elif e.get(4): info = msg(e[4][0]); kind = 'mux'
        elif e.get(8): kind = 'planes'
        name = bytes(first(info, 1, b'')).decode() if info else '?'
        bi = base_info(first(info, 2, b'')) if info else {}
        out.append(dict(direction=direction, conn=conn, kind=kind, name=name,
                        connected_context_index=first(csi, 4), connected_contexts=connected,
                        buffers=first(csi, 6), **bi))
    return out

def ops_config_bytes(ops) -> tuple[collections.Counter, collections.Counter]:
    """sum of write_data_ccw_ptr.size (16) and write_data_ccw.data (3) per cfg channel"""
    per_ch = collections.Counter(); n = collections.Counter()
    for op in ops:
        for a in msg(op)[2]:
            am = msg(a)
            for p in am.get(16, []):
                pm = msg(p); per_ch[first(pm, 3)] += first(pm, 2); n['ccw_ptr'] += 1
            for p in am.get(3, []):
                pm = msg(p); per_ch[first(pm, 2)] += len(first(pm, 1, b'')); n['ccw_inline'] += 1
            for p in am.get(4, []):
                n['compressed'] += 1
    return per_ch, n

def main(path: str) -> None:
    """Decode the HEF at path and print the byte listing to stdout."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    magic, version, proto_size = struct.unpack_from('>III', raw, 0)
    assert magic == 0x01484546, hex(magic)
    hdr = {0: 12+20, 1: 12+16, 2: 12+32, 3: 12+44}[version]
    print(f'# HEF {path}\n# header: magic=0x{magic:08x} version={version} proto_size={proto_size} header_size={hdr}')
    if version == 2:
        ccws_size = struct.unpack_from('>Q', raw, 12+8)[0]
        print(f'# v2 ccws_size={ccws_size} (file size {len(raw)} = header {hdr} + proto {proto_size} + ccws {ccws_size}'
              f' -> {hdr+proto_size+ccws_size})')
    hef = msg(memoryview(raw)[hdr:hdr+proto_size])
    for ng in hef[2]:
        g = msg(ng)
        name = bytes(first(msg(first(g, 1, b'')), 1, b'')).decode()
        print(f'\n## network_group {name}')
        core_ops = []
        for op in g.get(8, []):
            om = msg(op)
            if om.get(4):
                co = msg(om[4][0])
                print(f'op {bytes(first(om,1,b"")).decode()}: core_op contexts={len(co[3])} partial_core_ops={len(co[7])}')
                if co[3]: core_ops.append(('core_op', co))
                for pc in co[7]:
                    pm = msg(pc); lay = msg(first(pm, 2, b''))
                    core_ops.append((f'partial(layout_bitmap=0x{first(lay,1):x})', msg(first(pm, 1, b''))))
            elif om.get(5): print(f'op {bytes(first(om,1,b"")).decode()}: nms_op (host post-process)')
        if g.get(3): core_ops.insert(0, ('legacy_network_group', g))
        print('core op variants:', [n for n, _ in core_ops])
        label, g = core_ops[0]
        print(f'using variant: {label}')
        pre = msg(first(g, 2, b''))
        ch, n = ops_config_bytes(pre[1])
        print(f'preliminary_config: config_bytes_per_cfg_channel={dict(ch)} total={sum(ch.values())} actions={dict(n)}')
        totals = collections.Counter()
        for c in g[3]:
            cm = msg(c)
            idx = first(cm, 1)
            meta = first(cm, 3, b'')
            cname = bytes(first(msg(meta), 1, b'')).decode()
            ch, n = ops_config_bytes(cm[2])
            print(f'\n### context {idx} ({cname}) config_bytes_per_cfg_channel={dict(ch)} total={sum(ch.values())} actions={dict(n)}')
            totals['config'] += sum(ch.values())
            print('dir conn          kind name                                  HxWxF (padded)            db cbpb  cbpf  frame_B   padded_B  conn_ctx')
            for e in edge_layers(meta):
                fs = align8(e['height']*e['width']*e['features']*max(e['data_bytes'],1))
                ps = align8(e['padded_height']*e['padded_width']*e['padded_features']*max(e['data_bytes'],1))
                cb = e['core_bytes_per_buffer']*e['core_buffers_per_frame']
                print(f"{e['direction']} {e['conn']:13s} {e['kind']:5s} {e['name'][:37]:37s} "
                      f"{e['height']}x{e['width']}x{e['features']} ({e['padded_height']}x{e['padded_width']}x{e['padded_features']}) "
                      f"{e['data_bytes']:2d} {e['core_bytes_per_buffer']:5d} {e['core_buffers_per_frame']:5d} {fs:9d} {ps:9d} "
                      f"cbpb*cbpf={cb} {e['connected_context_index']} {e['connected_contexts']} bufs={e['buffers']}")
                totals[f"{e['conn']}_{e['direction']}"] += fs
                totals[f"{e['conn']}_{e['direction']}_padded"] += ps
                totals[f"{e['conn']}_{e['direction']}_cbpbxcbpf"] += cb
        print('\n## totals over dynamic contexts (bytes; edge layers per frame, config per batch)')
        for k in sorted(totals): print(f'{k:40s} {totals[k]:12d}')

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description='Decode a HEF protobuf and list per-context config (CCW) bytes and edge-layer '
                    'per-frame transfer sizes. Output goes to stdout.')
    ap.add_argument('hef', help='path to the .hef file (echoed as given in the output header)')
    return ap.parse_args(argv)

if __name__ == '__main__':
    main(parse_args(sys.argv[1:]).hef)
