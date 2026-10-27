#!/usr/bin/env python3
"""Aggregate preproc_cmp_run.sh output into results_raw.csv, results.csv and a plain-text table (stdout).

Inputs: <out_dir>/raw/<cfg>_run<N>/{stdout.log,run_env.txt} ([RESULT] key=value line of yvo_preproc_bench --impl) and
<out_dir>/evidence/accuracy/<name>.log ([ACCURACY] line). results.csv = per setting, median over runs of every
per-run value (per-run values are themselves medians / means over the timed frames), plus min/max of wall_med and
cpu_mean. Usage: python3 preproc_cmp_aggregate.py <out_dir>
"""
import collections
import csv
import re
import statistics
import sys
from pathlib import Path

TEXT_KEYS = ("impl", "cpu_out", "cpu_map")


def parse_kv(line):
    return dict(tok.split("=", 1) for tok in line.split()[1:] if "=" in tok)


def num(v):
    try:
        return float(v)
    except ValueError:
        return None


def main():
    out = Path(sys.argv[1])
    rows, order = [], []
    for log in sorted((out / "raw").glob("*_run*/stdout.log")):
        cfg, run = log.parent.name.rsplit("_run", 1)
        res = [ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("[RESULT]")]
        if not res:
            print(f"warning: no [RESULT] in {log}", file=sys.stderr)
            continue
        kv = parse_kv(res[-1])
        env = (log.parent / "run_env.txt").read_text().split() if (log.parent / "run_env.txt").exists() else []
        envd = {env[i]: env[i + 1] for i in range(0, len(env) - 1, 2)}
        row = {"config": cfg, "run": int(run), **kv,
               "soc_pre": envd.get("soc_pre", ""), "soc_post": envd.get("soc_post", ""),
               "cpus": envd.get("cpus", ""), "governor": envd.get("governor", "")}
        rows.append(row)
        if cfg not in order:
            order.append(cfg)
    keys = ["config", "run"]
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "results_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in sorted(rows, key=lambda r: (order.index(r["config"]), r["run"])):
            w.writerow({k: r.get(k, "") for k in keys})

    by = collections.defaultdict(list)
    for r in rows:
        by[r["config"]].append(r)
    num_keys = [k for k in keys if k not in ("config", "run", "cpus", "governor") + TEXT_KEYS]
    agg_keys = ["config", "runs", *TEXT_KEYS, "cpus", *num_keys, "wall_med_min", "wall_med_max",
                "cpu_mean_min", "cpu_mean_max"]
    agg = []
    for cfg in order:
        rs = by[cfg]
        a = {"config": cfg, "runs": len(rs), "cpus": rs[0].get("cpus", "")}
        for k in TEXT_KEYS:
            a[k] = rs[0].get(k, "")
        for k in num_keys:
            vals = [num(r[k]) for r in rs if k in r and num(r[k]) is not None]
            if not vals:
                a[k] = ""
            elif all(v.is_integer() for v in vals) and k in ("threads", "mem_threads", "frames", "warmup"):
                a[k] = str(int(statistics.median(vals)))
            else:
                a[k] = f"{statistics.median(vals):.4f}"
        for k in ("wall_med", "cpu_mean"):
            vals = [num(r[k]) for r in rs if k in r]
            a[f"{k}_min"] = f"{min(vals):.4f}" if vals else ""
            a[f"{k}_max"] = f"{max(vals):.4f}" if vals else ""
        agg.append(a)
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=agg_keys, lineterminator="\n")
        w.writeheader()
        for a in agg:
            w.writerow({k: a.get(k, "") for k in agg_keys})

    print("# Median over runs (per-run: median/p90/mean over the timed frames). ms per frame.")
    hdr = ("config", "runs", "wall_med", "wall_p90", "range(wall_med)", "cpu_mean", "range(cpu_mean)", "user", "sys",
           "map", "flatten", "resize", "pad", "transpose", "sync", "submit", "wait", "gpu")
    print("\t".join(hdr))
    for a in agg:
        g = a.get
        print("\t".join([a["config"], str(a["runs"]), g("wall_med", ""), g("wall_p90", ""),
                         f"{g('wall_med_min')}..{g('wall_med_max')}", g("cpu_mean", ""),
                         f"{g('cpu_mean_min')}..{g('cpu_mean_max')}", g("user_mean", ""), g("sys_mean", ""),
                         g("map_med", ""), g("flatten_med", ""), g("resize_med", ""), g("pad_med", ""),
                         g("transpose_med", ""), g("sync_med", ""), g("submit_med", ""), g("wait_med", ""),
                         g("gpu_med", "")]))

    acc_dir = out / "evidence" / "accuracy"
    if acc_dir.is_dir():
        print("\n# Accuracy vs CPU-ref (1 thread), every frame of the pass. content = 640x360x3 picture bytes, "
              "whole = 640x640x3.")
        print("name\tframes\tidentical_frames\tcontent_identical\tcontent_max_abs\tcontent_mean_abs\tcontent_psnr_db\t"
              "whole_psnr_db\thash\tflatten_vs_ref_desand_diff_bytes")
        for log in sorted(acc_dir.glob("*.log")):
            line = next((ln for ln in log.read_text(errors="replace").splitlines() if ln.startswith("[ACCURACY]")),
                        None)
            if not line:
                print(f"{log.stem}\t(no [ACCURACY] line)")
                continue
            m = re.search(r"content: (.*?) whole: (.*?) hash=(\w+)(.*)$", line)
            c = dict(t.split("=", 1) for t in m.group(1).split())
            wh = dict(t.split("=", 1) for t in m.group(2).split())
            kv = parse_kv(line)
            flat = re.search(r"flatten_vs_ref_desand_diff_bytes=(\d+)", m.group(4))
            print("\t".join([log.stem, kv.get("frames", ""), kv.get("identical_frames", ""), c["identical"],
                             c["max_abs"], c["mean_abs"], c["psnr_db"], wh["psnr_db"], m.group(3),
                             flat.group(1) if flat else "-"]))


if __name__ == "__main__":
    main()
