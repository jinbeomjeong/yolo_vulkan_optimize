#!/usr/bin/env python3
"""PCIe byte accounting for yolov11m.hef + fit of transfer models to fw-actions context times and T(B).
Inputs: HEF (decoded with hef_bytes.py), fwactions_probe_b{1,8}.json. Stdlib only.

Origin: written for the 07_results/candidate_reduction_survey experiment (2026-10-08) to test whether
shrinking the boundary outputs (fewer NMS candidates) could raise NPU throughput over PCIe Gen3 x1.
Its output was saved as 07_results/candidate_reduction_survey/evidence/pcie_byte_model_fit.txt
(sections 1-9; the "## 10. Notes" section there was written by hand). Originally named byte_model.py.

What it computes:
  1-2. per-context config and edge bytes from the HEF (via hef_bytes.py), per frame and per batch B
  3.   per-context spans and cfg descriptor sums from measure-fw-actions JSON (batch 1 and 8)
  4.   least-squares fit of t_ctx(B) = (cfg_ctx + B * X_ctx) / BW for three transfer models
  5.   the same models against measured T(B)
  6-7. upper bound of the FPS gain if D2H BOUNDARY bytes went to 0 (incl. a partial-overlap model)
  8-9. Gen2 cross-check and PCIe line-rate reference

Inputs:
  EVIDENCE_DIR  folder holding fwactions_probe_b1.json and fwactions_probe_b8.json
                (07_results/candidate_reduction_survey/evidence/). It is only read, never written.
  HEF           the .hef file (03_models/yolov11m.hef).
Hard-coded data (not read from files): the measured T(B) points in section 5
(07_results/npu_benchmark/runs/gen3 b1, 07_results/op_point/runs/latency_gen3_cooled b2/b4/b8), the Gen2
benchmark FPS in section 8 (07_results/npu_benchmark/runs/gen2), and the report title, which always names
03_models/yolov11m.hef whatever HEF is passed.

Output: stdout by default, or --out PATH. A path inside EVIDENCE_DIR is refused so archived results are
never overwritten.

Usage:
  python3 06_tools/python/hef_byte_model.py 07_results/candidate_reduction_survey/evidence \\
      03_models/yolov11m.hef --out build/out/hef_tools/pcie_byte_model_fit.txt
"""
import argparse
import collections
import json
import os
import struct
import sys
from pathlib import Path

# Make `import hef_bytes` work even when the script dir is not on sys.path (e.g. python3 -I).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hef_bytes as hb


def load_hef(path: str) -> tuple[int, list[dict]]:
    """Return (preliminary config bytes, per-dynamic-context {idx, cfg, h2d, d2h})."""
    with open(path, 'rb') as fh:
        raw = fh.read()
    _, ver, psz = struct.unpack_from('>III', raw, 0)
    hdr = {0: 32, 1: 28, 2: 44, 3: 56}[ver]
    hef = hb.msg(memoryview(raw)[hdr:hdr + psz])
    g = hb.msg(hef[2][0])
    for op in g[8]:
        om = hb.msg(op)
        if om.get(4): co = hb.msg(om[4][0]); break
    pre_cfg = sum(hb.ops_config_bytes(hb.msg(hb.first(co, 2, b''))[1])[0].values())
    ctxs = []
    for c in co[3]:
        cm = hb.msg(c)
        cfg = sum(hb.ops_config_bytes(cm[2])[0].values())
        h2d = collections.Counter(); d2h = collections.Counter()
        for e in hb.edge_layers(hb.first(cm, 3, b'')):
            # HailoRT accounting (matches measure-nnc-performance total exactly): core_bytes_per_buffer * core_buffers_per_frame
            b = e['core_bytes_per_buffer'] * e['core_buffers_per_frame']
            (h2d if e['direction'] == 'H2D' else d2h)[e['conn']] += b
        ctxs.append(dict(idx=hb.first(cm, 1), cfg=cfg, h2d=h2d, d2h=d2h))
    return pre_cfg, ctxs

def ctx_times(path: str) -> tuple[int, float, list[dict], collections.Counter]:
    """Read a measure-fw-actions JSON: (batch_size, fps, per-dynamic-context spans, preliminary descs)."""
    with open(path, encoding='utf-8') as fh:
        d = json.load(fh)
    r = d['runs'][0]
    clk = d['clock_cycle_MHz'] * 1e3   # cycles per ms
    dyn = [c for c in r['contexts'] if c['context_name'].startswith('dynamic_')]
    out = []
    for i, c in enumerate(dyn):
        ts = [a['timestamp'] for a in c['actions'] if 'timestamp' in a]
        span = (max(ts) - min(ts)) / clk
        nxt = None
        if i + 1 < len(dyn):
            ts2 = [a['timestamp'] for a in dyn[i + 1]['actions'] if 'timestamp' in a]
            nxt = (min(ts2) - min(ts)) / clk
        descs = collections.Counter()
        for a in c['actions']:
            if a['type'] == 'fetch_cfg_channel_descriptors':
                descs[a['data']['channel_index']] += a['data']['descriptors_count']
        out.append(dict(name=c['context_name'], span=span, start_to_next=nxt, descs=descs))
    pre = [c for c in r['contexts'] if c['context_name'] == 'preliminary'][0]
    pdescs = collections.Counter()
    for a in pre['actions']:
        if a['type'] == 'fetch_cfg_channel_descriptors':
            pdescs[a['data']['channel_index']] += a['data']['descriptors_count']
    return r['batch_size'], r['fps'], out, pdescs

def lsq_scale(xs, ys):            # y = x / BW  -> 1/BW = sum(xy)/sum(xx)
    k = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    return k

def lsq_affine(xs, ys):           # y = a + k x
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    k = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - k * mx, k

def build_report(EV: str, HEF: str) -> str:
    """Compute the full report (sections 1-9) and return it as text ending with a newline."""
    pre_cfg, ctxs = load_hef(HEF)
    L = []
    p = L.append
    MB = 1e6
    p('# PCIe byte accounting + transfer-model fit for 03_models/yolov11m.hef (HailoRT 4.24.0, Hailo-8, PCIe Gen3 x1)')
    p('# Generated by the stdlib decoder (hef_proto_bytes.txt has the decoder listing). Units: bytes, ms, GB/s = 1e9 B/s.')
    p('')
    p('## 1. Per-context bytes from HEF (config = sum of write_data_ccw_ptr.size; edges = core_bytes_per_buffer*core_buffers_per_frame)')
    p(f'preliminary config (once per activation, not per batch): {pre_cfg}')
    hdr = f"{'ctx':>4} {'cfg/batch':>10} | {'H2D bnd':>9} {'H2D IC':>9} {'H2D DDR':>9} {'H2D sum':>9} | {'D2H bnd':>9} {'D2H IC':>9} {'D2H DDR':>9} {'D2H sum':>9} | {'max(H,D)':>9} {'H+D':>9}"
    p(hdr)
    tot = collections.Counter()
    for c in ctxs:
        h = sum(c['h2d'].values()); d = sum(c['d2h'].values())
        c['H'] = h; c['D'] = d
        p(f"{c['idx']:>4} {c['cfg']:>10} | {c['h2d']['BOUNDARY']:>9} {c['h2d']['INTER_CONTEXT']:>9} {c['h2d']['DDR']:>9} {h:>9} | "
          f"{c['d2h']['BOUNDARY']:>9} {c['d2h']['INTER_CONTEXT']:>9} {c['d2h']['DDR']:>9} {d:>9} | {max(h, d):>9} {h + d:>9}")
        for k in ('BOUNDARY', 'INTER_CONTEXT', 'DDR'):
            tot['H2D_' + k] += c['h2d'][k]; tot['D2H_' + k] += c['d2h'][k]
        tot['cfg'] += c['cfg']; tot['H'] += h; tot['D'] += d; tot['max'] += max(h, d)
    p(f"{'sum':>4} {tot['cfg']:>10} | {tot['H2D_BOUNDARY']:>9} {tot['H2D_INTER_CONTEXT']:>9} {tot['H2D_DDR']:>9} {tot['H']:>9} | "
      f"{tot['D2H_BOUNDARY']:>9} {tot['D2H_INTER_CONTEXT']:>9} {tot['D2H_DDR']:>9} {tot['D']:>9} | {tot['max']:>9} {tot['H'] + tot['D']:>9}")
    S = tot['cfg'] + tot['H'] + tot['D']
    p(f'HailoRT get_total_transfer_size() equivalent (cfg + all edges, "per frame"): {S}')
    p('')
    p('## 2. Per batch of B frames (config once per batch, edges per frame)')
    p(f"{'B':>3} {'H2D bytes':>12} {'D2H bytes':>12} {'total':>12} {'per frame':>12}")
    for B in (1, 2, 4, 8):
        H = tot['cfg'] + B * tot['H']; D = B * tot['D']
        p(f"{B:>3} {H:>12} {D:>12} {H + D:>12} {(H + D) / B:>12.0f}")
    p('')

    # fw-actions
    p('## 3. fw-actions per-context times (200 MHz timestamps) and config descriptor sums')
    runs = {}
    for b in (1, 8):
        B, fps, cts, pdescs = ctx_times(os.path.join(EV, f'fwactions_probe_b{b}.json'))
        runs[B] = cts
        p(f'batch {B}: run fps {fps:.2f}; preliminary descs {dict(pdescs)} sum {sum(pdescs.values())} x512 = {512 * sum(pdescs.values())} (HEF preliminary cfg {pre_cfg})')
        for c, h in zip(cts, ctxs):
            nd = sum(c['descs'].values())
            p(f"  {c['name']}: span {c['span']:.2f} ms, start->next {c['start_to_next'] if c['start_to_next'] is None else round(c['start_to_next'], 2)} ms, "
              f"cfg descs {dict(c['descs'])} sum {nd} x512 = {nd * 512} (HEF cfg {h['cfg']}, ratio {nd * 512 / h['cfg']:.4f})")
        p(f"  dynamic span sum {sum(c['span'] for c in cts):.2f} ms; batch period from fps {1000 * B / fps:.2f} ms")
    p('')
    p('## 4. Model fit: t_ctx(B) = (cfg_ctx + B * X_ctx) / BW, one BW for all 8 points (4 contexts x B in {1,8})')
    models = {
        'M1 duplex: X = max(H2D,D2H)': lambda c: max(c['H'], c['D']),
        'M2 serial: X = H2D + D2H': lambda c: c['H'] + c['D'],
        'M3 H2D only: X = H2D': lambda c: c['H'],
    }
    best = None
    for name, fx in models.items():
        xs = []; ys = []
        for B in (1, 8):
            for c, h in zip(runs[B], ctxs):
                xs.append((h['cfg'] + B * fx(h)) / MB); ys.append(c['span'])
        k = lsq_scale(xs, ys)            # ms per MB
        bw = 1 / k                        # MB/ms = GB/s
        res = [y - k * x for x, y in zip(xs, ys)]
        rms = (sum(r * r for r in res) / len(res)) ** 0.5
        rel = max(abs(r) / y for r, y in zip(res, ys))
        p(f'{name}: BW_eff = {bw:.3f} GB/s, rms residual {rms:.2f} ms, max |rel| {100 * rel:.1f}%')
        p('   pred/meas ms: ' + ', '.join(f'{k * x:.1f}/{y:.1f}' for x, y in zip(xs, ys)))
        if best is None or rms < best[1]: best = (name, rms, bw, fx)
    p('')
    p('## 5. Whole-batch check against T(B) (07_results: gen3 benchmark b1, op_point/runs/latency_gen3_cooled b2/b4/b8)')
    TB = {1: 1000 / 24.3399, 2: 57.26, 4: 89.86, 8: 154.70}
    a, kf = lsq_affine(list(TB.keys()), list(TB.values()))
    p(f'measured T(B) fit: {a:.2f} + {kf:.2f} B ms (points {TB})')
    for name, fx in models.items():
        xs = [(tot['cfg'] + B * sum(fx(c) for c in ctxs)) / MB for B in TB]
        k = lsq_scale(xs, list(TB.values()))
        p(f'{name}: per-batch bytes = {tot["cfg"]/MB:.3f} + {sum(fx(c) for c in ctxs)/MB:.3f} B MB; '
          f'BW_eff {1 / k:.3f} GB/s -> predicted T(B) = {k * tot["cfg"] / MB:.2f} + {k * sum(fx(c) for c in ctxs) / MB:.2f} B ms')
    p('')
    p('## 6. Upper bound of throughput gain if boundary outputs (D2H BOUNDARY) are reduced to 0 bytes')
    for name, fx in models.items():
        for B in (1, 8):
            base = tot['cfg'] + B * sum(fx(c) for c in ctxs)
            cz = []
            for c in ctxs:
                c2 = dict(c); c2['D'] = c['D'] - c['d2h']['BOUNDARY']; cz.append(c2)
            new = tot['cfg'] + B * sum(fx(c) for c in cz)
            p(f'{name}, B={B}: bytes on critical path per batch {base} -> {new} ; FPS gain upper bound {100 * (base / new - 1):.2f}%')

    L2 = []
    q = L2.append
    q('')
    q('## 7. Partial-overlap model: X = max(H2D,D2H) + alpha*min(H2D,D2H); grid over alpha, BW by least squares (8 fw-actions points)')
    fits = []
    for i in range(0, 101):
        al = i / 100
        xs = []; ys = []
        for B in (1, 8):
            for c, h in zip(runs[B], ctxs):
                xs.append((h['cfg'] + B * (max(h['H'], h['D']) + al * min(h['H'], h['D']))) / MB); ys.append(c['span'])
        k = lsq_scale(xs, ys)
        rms = (sum((y - k * x) ** 2 for x, y in zip(xs, ys)) / len(xs)) ** 0.5
        fits.append((rms, al, 1 / k))
    fits.sort()
    for rms, al, bw in fits[:3]:
        q(f'alpha {al:.2f}: BW_eff {bw:.3f} GB/s, rms {rms:.2f} ms')
    rms_b, al_b, bw_b = fits[0]
    for al in sorted({0.0, al_b, 1.0}):
        for B in (1, 8):
            def xsum(cs): return sum(max(c['H'], c['D']) + al * min(c['H'], c['D']) for c in cs)
            cz = [dict(c, D=c['D'] - c['d2h']['BOUNDARY']) for c in ctxs]
            base = tot['cfg'] + B * xsum(ctxs); new = tot['cfg'] + B * xsum(cz)
            q(f'alpha {al:.2f} B={B}: critical bytes {base} -> {new}; FPS gain upper bound {100 * (base / new - 1):.2f}%')
    q('')
    q('## 8. Gen2 cross-check (07_results/npu_benchmark/runs/gen2 benchmark FPS b1/2/4/8; same bytes, BW should be ~half)')
    G2 = {1: 1000 / 12.7063, 2: 2000 / 18.0947, 4: 4000 / 22.6186, 8: 8000 / 26.0678}
    a2, k2 = lsq_affine(list(G2), list(G2.values()))
    q(f'measured Gen2 T(B) fit: {a2:.2f} + {k2:.2f} B ms')
    xs = [(tot['cfg'] + B * tot['max']) / MB for B in G2]
    k = lsq_scale(xs, list(G2.values()))
    q(f'M1 duplex on Gen2: BW_eff {1 / k:.3f} GB/s -> predicted {k * tot["cfg"] / MB:.2f} + {k * tot["max"] / MB:.2f} B ms')
    q('')
    q('## 9. Line-rate reference (not measured): PCIe Gen3 x1 = 8 GT/s * 128/130 / 8 = 0.985 GB/s per direction; Gen2 x1 = 5 GT/s * 8/10 / 8 = 0.500 GB/s')
    # The original wrote L (sections 1-6) and then appended L2 (sections 7-9); same bytes as one string.
    return '\n'.join(L) + '\n' + '\n'.join(L2) + '\n'

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description='PCIe byte accounting for a HEF and fit of transfer models to fw-actions context times '
                    'and measured T(B). Prints the report to stdout unless --out is given.')
    ap.add_argument('evidence_dir', help='folder with fwactions_probe_b1.json and fwactions_probe_b8.json '
                                         '(read only; nothing is written there)')
    ap.add_argument('hef', help='path to the .hef file')
    ap.add_argument('--out', metavar='PATH', help='write the report to PATH instead of stdout '
                                                  '(must not be inside evidence_dir)')
    args = ap.parse_args(argv)

    out = Path(args.out).resolve() if args.out else None
    if out is not None and out.is_relative_to(Path(args.evidence_dir).resolve()):
        ap.error(f'--out {args.out} is inside evidence_dir {args.evidence_dir}; refusing to write into the input folder')

    text = build_report(args.evidence_dir, args.hef)
    if out is None:
        sys.stdout.write(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', encoding='utf-8') as fh:
            fh.write(text)
    return 0

if __name__ == '__main__':
    sys.exit(main())
