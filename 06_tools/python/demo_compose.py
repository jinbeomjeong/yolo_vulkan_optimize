#!/usr/bin/env python3
"""Compose a before/after demo video from the logs of one real-time run. No GPU, no NPU; nothing here is measured.

The board has no H.264 hardware encoder, so recording while the pipeline runs would pollute its CPU figures.
06_tools/bench/demo_compare_bench.sh therefore only logs, and this tool replays the run afterwards:

  run folder (raw/<config>_run<N>/)   frame_log.csv, detections.jsonl, cpu_samples.csv, stdout.log; before: stage_log.csv
  ../../env_bench.txt                 board conditions for the footer line (optional)
  source video                        the same file the run decoded; "src" in the logs is its frame index

Time axis: output frame n stands for t = n/30 s on the frame log's clock. It shows the newest processed frame whose
present_ms is <= t, so dropped frames never appear and stutter is reproduced as it happened. Rows without present_ms in a
run that has present_ms never reached the screen (toolkit/frame_log.hpp) and are not shown; complete_ms replaces
present_ms only when the whole file has none (a run without display). Before the first present the picture is black.

Boxes: detections.jsonl "frame" is src and "box" is in source-frame pixels (inverse letterbox already applied by the tools).
Class names come from that file (core/coco_labels.hpp), box colors use class_color() of the same header.

On-screen text (Korean on purpose: the video is a talk asset):
  live values    refreshed every 0.5 s of output time from about the last second, "--" while there is not enough data
                 FPS        frames with complete_ms in (u - 1 s, u], divided by 1 s
                 CPU        pid_ticks delta of cpu_samples.csv between two sampler rows about 1 s apart, the newest not after
                            u; ms/frame divides by the frames completed between those two rows, % by their distance.
                            mono_ms = frame_log ms + mono_ms of the sampler's enable row (demo_compare_bench.sh)
  stage table    fixed run means. Each value and its origin is logged at startup. All are wall-clock ms per frame except
                 "before" decode with frame threads, which is the CPU time of the decoder threads per arrived frame
                 (the decode call only collects a finished frame then, see demo_compare_bench.sh).
                 after "GPU preprocess span" = mean(npu_done - submit) - mean(npu latency): submit_ms is taken before the
                 GPU submit. The "gpu shader ms" timestamp is not used: it is distorted on Vulkan composition + b1/s1
                 (07_results/input_pacing/runs/display_on/summary.md). after "GPU post" = mean(complete - npu_done), which
                 covers both GPU passes (decode + NMS share one wait, pipeline/gpu_postproc_stage.cpp).

Self-check (stderr, at the end): shown vs processed frames, OSD mean FPS/CPU vs stdout.log, hold-length histogram.
WARNING when FPS is off by 3 % or CPU by 10 % or more, or when the frame counts do not add up.
"""
import argparse
import bisect
import collections
import csv
import json
import logging
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("demo_compose")

OUT_W, OUT_H = 1920, 1080
OUT_FPS = 30
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_INDEX_KR = 1           # face order of both collections: JP, KR, SC, TC, HK (checked at startup)

LIVE_REFRESH_FRAMES = OUT_FPS // 2   # 2 updates per second
LIVE_WINDOW_MS = 1000.0
CPU_WINDOW_MIN_MS = 750.0            # sampler rows closer or farther apart than this are not "about one second"
CPU_WINDOW_MAX_MS = 1250.0
CPU_MAX_STALE_MS = 750.0             # newest sampler row older than this: sampling has ended or stalled

WARN_FPS_RATIO = 0.03
WARN_CPU_RATIO = 0.10
WARN_UNSHOWN_RATIO = 0.25            # presented frames lost between output ticks (30 fps output of a 60 Hz display)

TITLES = {
    "before": "적용 전 — 디코드·변환·결과 해석·표시 모두 CPU",
    "after": "적용 후 — 디코더·GPU·NPU 분담, CPU는 지휘만",
}
TOOL_HEADER = {"before": "== yvo_cpu_baseline", "after": "== yvo_pipeline"}
DEVICE_COLORS = {
    "CPU": (255, 170, 90),
    "NPU+CPU": (255, 214, 120),
    "NPU": (120, 200, 255),
    "GPU": (130, 230, 140),
    "HW 디코더": (215, 170, 255),
}


class ComposeError(RuntimeError):
    """The logs or the video cannot give a trustworthy result."""


# ── Logs ───────────────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FrameRow:
    src: int
    submit_ms: float | None
    npu_done_ms: float | None
    complete_ms: float | None
    present_ms: float | None


def _opt_float(text: str | None) -> float | None:
    if text is None or text == "" or text == "nan":
        return None
    return float(text)


def load_frame_log(path: Path) -> list[FrameRow]:
    """Processed frames (dropped == 0) in file order."""
    rows: list[FrameRow] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = {"src", "dropped", "submit_ms", "npu_done_ms", "complete_ms"} - set(reader.fieldnames or [])
        if missing:
            raise ComposeError(f"{path}: missing columns {sorted(missing)}")
        for r in reader:
            if r["dropped"] != "0":
                continue
            rows.append(FrameRow(int(r["src"]), _opt_float(r["submit_ms"]), _opt_float(r["npu_done_ms"]),
                                 _opt_float(r["complete_ms"]), _opt_float(r.get("present_ms"))))
    if not rows:
        raise ComposeError(f"{path}: no processed frames")
    return rows


def screen_events(rows: list[FrameRow]) -> tuple[list[tuple[float, int]], str]:
    """(time on screen in ms, src) in time order, and the name of the column used."""
    if any(r.present_ms is not None for r in rows):
        column = "present_ms"
        events = [(r.present_ms, r.src) for r in rows if r.present_ms is not None]
    else:
        column = "complete_ms"
        events = [(r.complete_ms, r.src) for r in rows if r.complete_ms is not None]
    events.sort()
    for (_, a), (_, b) in zip(events, events[1:]):
        if b <= a:   # the source video is only read forward, and a display never goes back in time
            raise ComposeError(f"{column} is not in src order: src {b} follows src {a}")
    if not events:
        raise ComposeError("no frame ever reached the screen")
    return events, column


def iter_detections(path: Path) -> Iterator[tuple[int, list[dict]]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            yield obj["frame"], obj["detections"]


class DetectionReader:
    """Forward-only lookup in detections.jsonl."""

    def __init__(self, path: Path) -> None:
        self._it = iter_detections(path)
        self._next: tuple[int, list[dict]] | None = next(self._it, None)
        self.missing = 0

    def get(self, src: int) -> list[dict]:
        while self._next is not None and self._next[0] < src:
            self._next = next(self._it, None)
        if self._next is None or self._next[0] != src:
            self.missing += 1   # a presented frame without a line: drawn without boxes and reported at the end
            return []
        return self._next[1]


@dataclass(frozen=True)
class CpuSamples:
    times_ms: list[float]   # frame log clock (0 = enable row)
    ticks: list[int]
    ms_per_tick: float
    enable_mono_ms: float


def load_cpu_samples(path: Path) -> CpuSamples:
    enable: float | None = None
    raw: list[tuple[float, int]] = []
    clk_tck = 0
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            mono = float(r["mono_ms"])
            clk_tck = int(r["clk_tck"])
            if r["event"] == "enable" and enable is None:
                enable = mono
            if r["pid_ticks"]:
                raw.append((mono, int(r["pid_ticks"])))
    if enable is None:
        raise ComposeError(f"{path}: no enable row, the sampler never saw the tool's t0")
    if clk_tck <= 0:
        raise ComposeError(f"{path}: no clk_tck")
    kept = [(m - enable, t) for m, t in raw if m >= enable]
    return CpuSamples([k[0] for k in kept], [k[1] for k in kept], 1000.0 / clk_tck, enable)


class LiveValues:
    """The live OSD numbers at output time u. None = not enough data, shown as "--"."""

    def __init__(self, rows: list[FrameRow], cpu: CpuSamples) -> None:
        self._complete = sorted(r.complete_ms for r in rows if r.complete_ms is not None)
        self._cpu = cpu

    def _frames_in(self, t0_ms: float, t1_ms: float) -> int:
        return bisect.bisect_right(self._complete, t1_ms) - bisect.bisect_right(self._complete, t0_ms)

    def fps(self, u_ms: float) -> float | None:
        if u_ms < LIVE_WINDOW_MS:
            return None
        return self._frames_in(u_ms - LIVE_WINDOW_MS, u_ms) * 1000.0 / LIVE_WINDOW_MS

    def cpu(self, u_ms: float) -> tuple[float | None, float | None]:
        """(ms per processed frame, % of one core), both from one pair of sampler rows."""
        times = self._cpu.times_ms
        i1 = bisect.bisect_right(times, u_ms) - 1
        if i1 < 1 or u_ms - times[i1] > CPU_MAX_STALE_MS:
            return None, None
        i0 = min(range(i1), key=lambda i: abs(times[i] - (times[i1] - LIVE_WINDOW_MS)))
        span = times[i1] - times[i0]
        if not CPU_WINDOW_MIN_MS <= span <= CPU_WINDOW_MAX_MS:
            return None, None
        cpu_ms = (self._cpu.ticks[i1] - self._cpu.ticks[i0]) * self._cpu.ms_per_tick
        frames = self._frames_in(times[i0], times[i1])
        return (cpu_ms / frames if frames else None), 100.0 * cpu_ms / span


@dataclass(frozen=True)
class Summary:
    """The lines of stdout.log the self-check and the stage table need. None = the line is not there."""

    text: str
    frames: int | None
    fps: float | None
    cpu_ms_per_frame: float | None
    cpu_core_pct: float | None
    t0_monotonic_ms: float | None

    def mean(self, label: str) -> float | None:
        m = re.search(rf"^{re.escape(label)}\s+mean ([\d.]+)", self.text, re.M)
        return float(m.group(1)) if m else None


def load_summary(path: Path, label: str) -> Summary:
    text = path.read_text(encoding="utf-8", errors="replace")
    if TOOL_HEADER[label] not in text:
        raise ComposeError(f"{path}: \"{TOOL_HEADER[label]}\" not found, is this really a --label {label} run?")

    def number(pattern: str) -> float | None:
        m = re.search(pattern, text, re.M)
        return float(m.group(1)) if m else None

    frames = number(r"^frames\s+(\d+)")
    return Summary(text, int(frames) if frames is not None else None, number(r"^wall\s+[\d.]+ s, ([\d.]+) FPS"),
                   number(r"^cpu\s+([\d.]+) ms/frame"), number(r"^cpu\s+.*?, (\d+)% of one core"),
                   number(r"^t0_monotonic_ms\s+([\d.]+)"))


@dataclass(frozen=True)
class StageRow:
    key: str               # for the log
    label: str             # on screen
    device: str
    value_ms: float | None
    origin: str            # for the log


def _csv_means(path: Path, skip_first_src_for: tuple[str, ...] = ()) -> dict[str, float]:
    """Column means of stage_log.csv. Empty cells are skipped. Rows with src 0 are skipped for the given columns."""
    values: dict[str, list[float]] = collections.defaultdict(list)
    with path.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            for k, v in r.items():
                if k != "src" and v and not (k in skip_first_src_for and r["src"] == "0"):
                    values[k].append(float(v))
    return {k: sum(v) / len(v) for k, v in values.items()}


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stage_rows_before(run_dir: Path, summary: Summary) -> list[StageRow]:
    # Frame 0 is decoded before t0 and logged as 0.000 (cpu_baseline_pipeline.cpp), so it is not a decode time.
    means = _csv_means(run_dir / "stage_log.csv", skip_first_src_for=("decode_ms",))
    for col in ("decode_ms", "swscale_ms", "npu_ms", "disp_convert_ms", "disp_overlay_ms"):
        if col not in means:
            raise ComposeError(f"{run_dir / 'stage_log.csv'}: column {col} has no values")
    m = re.search(r"^\[decoder\].* threads=(\d+) thread_type=(\w+)", summary.text, re.M)
    if not m:
        raise ComposeError("stdout.log: no [decoder] line, cannot tell what the decode time means")
    if m.group(2) == "frame":
        t = re.search(r"^thread cpu\s+av:hevc:df\s+threads\s+\d+\s+[\d.]+ ms\s+([\d.]+) ms/arrived", summary.text, re.M)
        decode = StageRow("decode", "디코드 (스레드 CPU 합)", "CPU", float(t.group(1)) if t else None,
                          f"stdout.log 'thread cpu av:hevc:df' ms/arrived, {m.group(1)} frame threads: "
                          "CPU time, not wall clock (the decode call only collects a frame)")
    else:
        decode = StageRow("decode", "디코드", "CPU", means["decode_ms"],
                          f"stage_log.csv mean(decode_ms) over arrived frames without src 0, thread_type={m.group(2)}")
    return [
        decode,
        StageRow("swscale", "변환 (swscale)", "CPU", means["swscale_ms"], "stage_log.csv mean(swscale_ms)"),
        StageRow("npu", "NPU + HailoRT 변환·NMS", "NPU+CPU", means["npu_ms"],
                 "stage_log.csv mean(npu_ms): submit -> completion callback, HailoRT transpose + NPU + host NMS"),
        StageRow("display_prep", "표시 준비", "CPU", means["disp_convert_ms"] + means["disp_overlay_ms"],
                 "stage_log.csv mean(disp_convert_ms) + mean(disp_overlay_ms)"),
    ]


def _sum_or_none(a: float | None, b: float | None) -> float | None:
    return a + b if a is not None and b is not None else None


def stage_rows_after(rows: list[FrameRow], summary: Summary) -> list[StageRow]:
    npu = summary.mean("npu latency ms")
    span = _mean([r.npu_done_ms - r.submit_ms for r in rows if r.npu_done_ms is not None and r.submit_ms is not None])
    post = _mean([r.complete_ms - r.npu_done_ms for r in rows if r.complete_ms is not None and r.npu_done_ms is not None])
    return [
        StageRow("decode", "디코드 호출 (하드웨어)", "HW 디코더", summary.mean("decode call ms"),
                 "stdout.log 'decode call ms' mean over arrived frames"),
        StageRow("gpu_pre", "GPU 전처리 구간", "GPU", span - npu if span is not None and npu is not None else None,
                 "frame_log.csv mean(npu_done_ms - submit_ms) - stdout.log 'npu latency ms' mean: wall clock from before "
                 "the GPU submit to the NPU submit"),
        StageRow("npu", "NPU", "NPU", npu, "stdout.log 'npu latency ms' mean: NPU submit -> completion callback"),
        StageRow("gpu_post", "GPU 후처리 (NMS 포함)", "GPU", post,
                 "frame_log.csv mean(complete_ms - npu_done_ms): wall clock over both GPU passes, compare 'slot hold ms'"),
        StageRow("display", "표시 변환·오버레이", "GPU", _sum_or_none(summary.mean("disp convert ms"), summary.mean("disp overlay ms")),
                 "stdout.log 'disp convert ms' + 'disp overlay ms' means: GPU timestamps of the two display passes"),
    ]


def footer_lines(run_dir: Path, video: Path, video_info: dict) -> list[str]:
    """Only facts found in the video and in env_bench.txt; missing items are left out."""
    conditions = [f"입력 {video.name}", f"{video_info['width']}×{video_info['height']}", f"{video_info['fps']:g} fps"]
    env_path = run_dir.parent.parent / "env_bench.txt"
    if env_path.is_file():
        env = env_path.read_text(encoding="utf-8", errors="replace")
        arm = re.search(r"^frequency\(0\)=(\d+)", env, re.M)
        gov = re.search(r"^cpufreq policy0: governor (\S+)", env, re.M)
        if arm and gov:
            conditions.append(f"CPU {int(arm.group(1)) / 1e9:.1f} GHz ({gov.group(1)})")
        elif gov:
            conditions.append(f"CPU 거버너 {gov.group(1)}")
        v3d = re.search(r"^v3d_freq=(\d+)", env, re.M)
        if v3d:
            conditions.append(f"GPU V3D {v3d.group(1)} MHz")
        pcie = re.search(r"^pcie: .*?(x\d+) \((Gen\d)", env, re.M)
        if pcie:
            conditions.append(f"PCIe {pcie.group(2)} {pcie.group(1)}")
    else:
        log.warning("%s not found: board conditions left out of the footer", env_path)
    return ["실행 기록으로 합성한 영상 · 수치는 실제 실행에서 측정", " · ".join(conditions)]


# ── Video I/O ──────────────────────────────────────────────────────────────────────────────────
def probe_video(path: Path) -> dict:
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
         "stream=width,height,r_frame_rate,color_space,nb_read_packets", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8").stdout
    streams = json.loads(out)["streams"]
    if not streams:
        raise ComposeError(f"{path}: no video stream")
    s = streams[0]
    if (s["width"], s["height"]) != (OUT_W, OUT_H):
        raise ComposeError(f"{path}: {s['width']}x{s['height']}, the box mapping here only covers {OUT_W}x{OUT_H} sources")
    matrix = s.get("color_space", "unknown")
    if matrix not in ("bt709", "unknown"):
        raise ComposeError(f"{path}: color space {matrix}, only bt709 is handled")
    num, den = s["r_frame_rate"].split("/")
    return {"width": s["width"], "height": s["height"], "fps": int(num) / int(den), "packets": int(s["nb_read_packets"])}


class SourceReader:
    """Forward-only RGB frames from an ffmpeg pipe. Frame index = display order = src of the logs."""

    def __init__(self, path: Path) -> None:
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-an", "-sn",
               "-fps_mode", "passthrough",
               "-vf", "scale=in_color_matrix=bt709:in_range=tv:out_range=pc:flags=bicubic+accurate_rnd+full_chroma_int,"
                      "format=rgb24",
               "-f", "rawvideo", "pipe:1"]
        self._proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        self._size = OUT_W * OUT_H * 3
        self._buf = bytearray(self._size)
        self._view = memoryview(self._buf)
        self._index = -1

    def _read_one(self) -> None:
        got = 0
        while got < self._size:
            n = self._proc.stdout.readinto(self._view[got:])
            if not n:
                raise ComposeError(f"source video ended after {self._index + 1} frames, frame {self._index + 1} is needed")
            got += n
        self._index += 1

    def frame(self, src: int) -> np.ndarray:
        """Frame src as a writable view of the internal buffer, valid until the next call."""
        if src <= self._index:
            raise ComposeError(f"frame {src} requested after frame {self._index}: the reader only goes forward")
        while self._index < src:
            self._read_one()
        return np.frombuffer(self._buf, dtype=np.uint8).reshape(OUT_H, OUT_W, 3)

    def close(self) -> None:
        self._proc.terminate()   # drain so that ffmpeg stops on the signal, not on a broken pipe (which it reports as an error)
        while self._proc.stdout.read(1 << 20):
            pass
        self._proc.stdout.close()
        self._proc.wait()


class Encoder:
    """libx264 yuv420p through a pipe, into PATH.part.mp4 until it succeeded, so a failure leaves no plausible file."""

    def __init__(self, out: Path, crf: float, preset: str) -> None:
        self._out = out
        self._part = out.with_name(out.stem + ".part.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{OUT_W}x{OUT_H}", "-r", str(OUT_FPS), "-i", "pipe:0", "-an",
               "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709:flags=bicubic+accurate_rnd,format=yuv420p",
               "-c:v", "libx264", "-preset", preset, "-crf", f"{crf:g}", "-pix_fmt", "yuv420p",
               "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
               "-movflags", "+faststart", "-fflags", "+bitexact", str(self._part)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, bufsize=0)

    def write(self, frame: np.ndarray) -> None:
        view = memoryview(frame).cast("B")
        while view:
            view = view[self._proc.stdin.write(view):]

    def finish(self) -> None:
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise ComposeError(f"ffmpeg encoder exited with {self._proc.returncode}")
        self._part.replace(self._out)

    def abort(self) -> None:
        self._proc.stdin.close()
        self._proc.terminate()
        self._proc.wait()
        self._part.unlink(missing_ok=True)


# ── Drawing ────────────────────────────────────────────────────────────────────────────────────
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size, index=FONT_INDEX_KR)
    if " KR" not in font.getname()[0]:
        raise ComposeError(f"{path} face {FONT_INDEX_KR} is {font.getname()[0]}, not the Korean one")
    return font


def class_color(class_id: int) -> tuple[int, int, int]:
    """core/coco_labels.hpp class_color(), RGB."""
    c = max(class_id, 0)
    return 255 - (c * 29) % 256, (c * 131 + 80) % 256, (c * 67) % 256


def blend(dst: np.ndarray, rgba: np.ndarray, x: int, y: int) -> None:
    """Integer alpha blend, in place. rgba must fit inside dst."""
    h, w = rgba.shape[:2]
    region = dst[y:y + h, x:x + w]
    a = rgba[..., 3:].astype(np.uint16)
    region[:] = ((rgba[..., :3].astype(np.uint16) * a + region.astype(np.uint16) * (255 - a) + 127) // 255).astype(np.uint8)


def fmt(value: float | None, digits: int) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


class Painter:
    """Boxes, labels and the panels. Everything is a pure function of its inputs, so the output is deterministic."""

    MARGIN = 24
    PAD = 18
    BOX_WIDTH = 3

    def __init__(self, label: str, stages: list[StageRow], footer: list[str]) -> None:
        self._title = TITLES[label]
        self._stages = stages
        self._f_title = load_font(FONT_BOLD, 30)
        self._f_head = load_font(FONT_BOLD, 21)
        self._f_small = load_font(FONT_REGULAR, 20)
        self._f_row = load_font(FONT_REGULAR, 23)
        self._f_value = load_font(FONT_BOLD, 46)
        self._f_box = load_font(FONT_BOLD, 22)
        self._f_foot = load_font(FONT_REGULAR, 20)
        self._labels: dict[tuple[str, int], np.ndarray] = {}
        self._footer = self._render_footer(footer)
        # The same panel size in both videos, so that they can be compared side by side.
        self._panel_w = max(self._f_title.getbbox(t)[2] for t in TITLES.values()) + 2 * self.PAD
        self._waiting = self._render_waiting()

    def _label_image(self, text: str, class_id: int) -> np.ndarray:
        key = (text, class_id)
        img = self._labels.get(key)
        if img is None:
            color = class_color(class_id)
            box = self._f_box.getbbox(text)
            pil = Image.new("RGB", (box[2] + 12, self._f_box.size + 10), color)
            luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
            ImageDraw.Draw(pil).text((6, pil.height // 2), text, font=self._f_box, anchor="lm",
                                     fill=(0, 0, 0) if luma > 140 else (255, 255, 255))
            img = np.asarray(pil, dtype=np.uint8)
            self._labels[key] = img
        return img

    def draw_detections(self, frame: np.ndarray, detections: list[dict]) -> None:
        clipped = []
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            x1, x2 = (int(round(min(max(v, 0.0), OUT_W - 1))) for v in (x1, x2))
            y1, y2 = (int(round(min(max(v, 0.0), OUT_H - 1))) for v in (y1, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), class_color(d["class"]), self.BOX_WIDTH, cv2.LINE_8)
            clipped.append((x1, y1, y2, d))
        placed: list[tuple[int, int, int, int, np.ndarray]] = []
        # After all rectangles, so that no rectangle crosses a label. The best score picks its place first and is drawn last.
        for x1, y1, y2, d in sorted(clipped, key=lambda c: -c[3]["score"]):
            img = self._label_image(f"{d['name']} {d['score']:.2f}", d["class"])
            h, w = img.shape[:2]
            lx = min(x1, OUT_W - w)
            first_y = y1 - h if y1 >= h else y1   # above the box, inside it at the top edge of the picture
            # Then under the box, then a little higher: a label stays next to its box or overlaps a lower priority.
            options = [first_y, y2 + 1, first_y - h, first_y - 2 * h]
            free = (c for c in options if 0 <= c <= OUT_H - h and not any(
                lx < px + pw and px < lx + w and c < py + ph and py < c + h for px, py, pw, ph, _ in placed))
            placed.append((lx, next(free, first_y), w, h, img))
        for lx, ly, w, h, img in reversed(placed):
            frame[ly:ly + h, lx:lx + w] = img

    def _render_footer(self, lines: list[str]) -> np.ndarray:
        width = max(self._f_foot.getbbox(s)[2] for s in lines) + 24
        line_h = self._f_foot.size + 8
        pil = Image.new("RGBA", (width, line_h * len(lines) + 12), (0, 0, 0, 150))
        draw = ImageDraw.Draw(pil)
        for i, s in enumerate(lines):
            draw.text((12, 6 + i * line_h), s, font=self._f_foot, fill=(235, 235, 235, 255))
        return np.asarray(pil, dtype=np.uint8)

    def _render_waiting(self) -> np.ndarray:
        text = "첫 처리 결과가 화면에 올라오기 전"
        box = self._f_row.getbbox(text)
        pil = Image.new("RGBA", (box[2] + 4, box[3] + 4), (0, 0, 0, 0))
        ImageDraw.Draw(pil).text((0, 0), text, font=self._f_row, fill=(200, 200, 200, 255))
        return np.asarray(pil, dtype=np.uint8)

    def render_panel(self, fps: float | None, cpu_ms: float | None, core_pct: float | None) -> np.ndarray:
        w = self._panel_w
        pad = self.PAD
        row_h = 33
        h = pad + 42 + 12 + 26 + 66 + 12 + 30 + row_h * len(self._stages) + pad - 4
        pil = Image.new("RGBA", (w, h), (0, 0, 0, 170))
        draw = ImageDraw.Draw(pil)
        y = pad - 6
        draw.text((pad, y), self._title, font=self._f_title, fill=(255, 255, 255, 255))
        y += 42 + 12
        draw.line((pad, y - 6, w - pad, y - 6), fill=(255, 255, 255, 90), width=1)

        live = [(0, "처리 FPS", fmt(fps, 1)),
                (150, "응용 CPU ms/프레임", fmt(cpu_ms, 1)),
                (345, "코어 사용률 % (1코어=100)", fmt(core_pct, 0))]
        for dx, name, value in live:
            draw.text((pad + dx, y), name, font=self._f_small, fill=(205, 205, 205, 255))
            draw.text((pad + dx, y + 20), value, font=self._f_value, fill=(255, 255, 255, 255))
        y += 26 + 66 + 12

        draw.line((pad, y - 8, w - pad, y - 8), fill=(255, 255, 255, 90), width=1)
        draw.text((pad, y - 2), "단계별 시간 · 실행 평균", font=self._f_head, fill=(255, 255, 255, 255))
        draw.text((w - pad, y - 2), "ms/프레임", font=self._f_small, anchor="ra", fill=(205, 205, 205, 255))
        y += 30
        for row in self._stages:
            draw.text((pad, y), row.label, font=self._f_row, fill=(245, 245, 245, 255))
            draw.text((w - pad - 90, y), row.device, font=self._f_row, anchor="ra",
                      fill=(*DEVICE_COLORS[row.device], 255))
            draw.text((w - pad, y), fmt(row.value_ms, 1), font=self._f_row, anchor="ra", fill=(255, 255, 255, 255))
            y += row_h
        return np.asarray(pil, dtype=np.uint8)

    def waiting_note(self, frame: np.ndarray) -> None:
        h, w = self._waiting.shape[:2]
        blend(frame, self._waiting, (OUT_W - w) // 2, (OUT_H - h) // 2)

    def overlay(self, frame: np.ndarray, panel: np.ndarray) -> None:
        blend(frame, panel, self.MARGIN, self.MARGIN)
        blend(frame, self._footer, self.MARGIN, OUT_H - self.MARGIN - self._footer.shape[0])


# ── Composition ────────────────────────────────────────────────────────────────────────────────
@dataclass
class RunStats:
    total_frames: int = 0
    black_frames: int = 0
    shown: list[int] | None = None
    fps_values: list[float] | None = None
    cpu_values: list[float] | None = None
    core_values: list[float] | None = None
    fps_dashes: int = 0
    cpu_dashes: int = 0


def output_frame_count(events: list[tuple[float, int]], max_seconds: float | None) -> int:
    """The first tick at or after the last present shows the last frame once, then the video ends."""
    last_ms = events[-1][0]
    total = int(np.ceil(last_ms * OUT_FPS / 1000.0)) + 1
    if max_seconds is not None:
        total = min(total, int(max_seconds * OUT_FPS))
    return total


def compose(args: argparse.Namespace, rows: list[FrameRow], events: list[tuple[float, int]], live: LiveValues,
            painter: Painter, detections: DetectionReader) -> RunStats:
    total = output_frame_count(events, args.max_seconds)
    dumps = sorted((int(float(t) * OUT_FPS + 1e-9), Path(p)) for t, p in args.dump_frame)
    if not args.out:
        total = min(total, dumps[-1][0] + 1)
    for n, p in dumps:
        if n >= total:
            raise ComposeError(f"--dump-frame {n / OUT_FPS:.3f} s ({p}) is past the end of the video ({total} frames)")

    source = SourceReader(args.video)
    encoder = Encoder(args.out, args.crf, args.preset) if args.out else None
    stats = RunStats(total, 0, [], [], [], [])
    base = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)   # picture without panels
    painter.waiting_note(base)
    final = np.zeros_like(base)
    panel = painter.render_panel(None, None, None)
    event_i = -1
    drawn_src: int | None = None
    dirty = True
    ok = False
    try:
        for n in range(total):
            t_ms = n * 1000.0 / OUT_FPS
            while event_i + 1 < len(events) and events[event_i + 1][0] <= t_ms:
                event_i += 1
            src = events[event_i][1] if event_i >= 0 else None
            if src != drawn_src:
                base = source.frame(src)
                painter.draw_detections(base, detections.get(src))
                drawn_src = src
                dirty = True
            if n % LIVE_REFRESH_FRAMES == 0:
                fps = live.fps(t_ms)
                cpu_ms, core_pct = live.cpu(t_ms)
                panel = painter.render_panel(fps, cpu_ms, core_pct)
                dirty = True
                if fps is None:
                    stats.fps_dashes += 1
                else:
                    stats.fps_values.append(fps)
                if cpu_ms is None:
                    stats.cpu_dashes += 1
                else:
                    stats.cpu_values.append(cpu_ms)
                if core_pct is not None:
                    stats.core_values.append(core_pct)
            if dirty:
                np.copyto(final, base)
                painter.overlay(final, panel)
                dirty = False
            if src is None:
                stats.black_frames += 1
            else:
                stats.shown.append(src)
            if encoder:
                encoder.write(final)
            for dn, dp in dumps:
                if dn == n:
                    dp.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(final).save(dp)
                    log.info("dump t=%.3f s (output frame %d): src %s -> %s", t_ms / 1000.0, n, src, dp)
            if n and n % 300 == 0:
                log.info("%d / %d output frames", n, total)
        ok = True
    finally:
        source.close()
        if encoder:
            encoder.finish() if ok else encoder.abort()
    return stats


def hold_histogram(shown: list[int]) -> collections.Counter:
    """How many output frames in a row showed the same source frame."""
    runs = collections.Counter()
    if shown:
        length = 1
        for a, b in zip(shown, shown[1:]):
            if a == b:
                length += 1
            else:
                runs[length] += 1
                length = 1
        runs[length] += 1
    return runs


def _compare(name: str, ours: float | None, theirs: float | None, limit: float) -> bool:
    """Log one comparison. Returns True when it passes or cannot be made."""
    if ours is None or theirs is None:
        log.warning("check %s: not available (ours %s, stdout.log %s)", name, ours, theirs)
        return True
    ratio = abs(ours - theirs) / theirs if theirs else 0.0
    passed = ratio < limit
    log.log(logging.INFO if passed else logging.WARNING, "check %s: OSD mean %.3f, stdout.log %.3f, off by %.1f%% (limit %.0f%%)",
            name, ours, theirs, 100.0 * ratio, 100.0 * limit)
    return passed


def self_check(args: argparse.Namespace, rows: list[FrameRow], events: list[tuple[float, int]], summary: Summary,
               cpu: CpuSamples, stats: RunStats, detections: DetectionReader, column: str) -> bool:
    end_ms = (stats.total_frames - 1) * 1000.0 / OUT_FPS
    partial = stats.total_frames < output_frame_count(events, None)
    good = True

    processed = sum(1 for r in rows if r.complete_ms is not None and r.complete_ms <= end_ms) if partial else len(rows)
    presented = sum(1 for t, _ in events if t <= end_ms)
    unique = len(set(stats.shown))
    never = len(rows) - len(events)
    between = presented - unique
    log.info("check frames%s: processed %d, reached the screen (%s) %d, shown in the video %d, lost between output ticks %d "
             "(%.1f%%), never on screen %d", " (--max-seconds window)" if partial else "", processed, column, presented,
             unique, between, 100.0 * between / presented if presented else 0.0, never)
    log.info("check output: %d frames = %.3f s, black lead-in %d", stats.total_frames, stats.total_frames / OUT_FPS,
             stats.black_frames)
    if between < 0 or (not partial and presented != len(events)):
        log.warning("check frames: the counts do not add up")
        good = False
    if presented and between / presented >= WARN_UNSHOWN_RATIO:
        log.warning("check frames: %.0f%% of the presented frames fall between output ticks", 100.0 * between / presented)
        good = False
    valid = {r.src for r in rows}
    alien = [s for s in stats.shown if s not in valid]
    if alien:
        log.warning("check frames: %d output frames show a frame that was not processed, first src %d", len(alien), alien[0])
        good = False
    if never:
        log.warning("check frames: %d processed frames have no %s and are not shown", never, column)
    if detections.missing:
        log.warning("check frames: %d presented frames have no line in detections.jsonl and were drawn without boxes",
                    detections.missing)
        good = False
    if not partial and summary.frames is not None and summary.frames != len(rows):
        log.warning("check frames: stdout.log says %d frames, frame_log.csv has %d processed", summary.frames, len(rows))
        good = False

    hist = hold_histogram(stats.shown)
    log.info("check hold: output frames per shown frame %s",
             ", ".join(f"{k}:{v}" for k, v in sorted(hist.items())) or "none")
    log.info("check OSD: %d FPS values (%d shown as --), %d CPU values (%d shown as --)", len(stats.fps_values),
             stats.fps_dashes, len(stats.cpu_values), stats.cpu_dashes)
    if summary.t0_monotonic_ms is not None:
        log.info("check time base: enable row - t0_monotonic_ms = %.3f ms", cpu.enable_mono_ms - summary.t0_monotonic_ms)
    if partial:
        log.info("check OSD means: skipped, stdout.log describes the whole run and --max-seconds cut it")
    else:
        good &= _compare("FPS", _mean(stats.fps_values), summary.fps, WARN_FPS_RATIO)
        good &= _compare("CPU ms/frame", _mean(stats.cpu_values), summary.cpu_ms_per_frame, WARN_CPU_RATIO)
        good &= _compare("CPU % of one core", _mean(stats.core_values), summary.cpu_core_pct, WARN_CPU_RATIO)
    return good


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compose a demo video from the logs of one real-time run "
                                            "(06_tools/bench/demo_compare_bench.sh).")
    p.add_argument("run_dir", type=Path, help="run folder: raw/<config>_run<N>/")
    p.add_argument("video", type=Path, help="the source video of that run")
    p.add_argument("--label", required=True, choices=("before", "after"), help="which config the run folder holds")
    p.add_argument("--out", type=Path, help="output mp4, 1920x1080 30 fps. May be left out with --dump-frame")
    p.add_argument("--crf", type=float, default=20.0, help="libx264 CRF (default 20)")
    p.add_argument("--preset", default="medium", help="libx264 preset (default medium)")
    p.add_argument("--max-seconds", type=float, help="stop after this much output, for trials")
    p.add_argument("--dump-frame", nargs=2, action="append", default=[], metavar=("T", "PATH"),
                   help="save the output frame at T seconds as PNG (repeatable)")
    args = p.parse_args(argv)
    if not args.out and not args.dump_frame:
        p.error("give --out, --dump-frame or both")
    if args.out and args.out.suffix != ".mp4":
        p.error("--out must end in .mp4")
    if args.max_seconds is not None and args.max_seconds <= 0:
        p.error("--max-seconds must be > 0")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s", stream=sys.stderr)
    args = parse_args(argv)
    try:
        run = args.run_dir
        rows = load_frame_log(run / "frame_log.csv")
        events, column = screen_events(rows)
        cpu = load_cpu_samples(run / "cpu_samples.csv")
        summary = load_summary(run / "stdout.log", args.label)
        video_info = probe_video(args.video)
        if events[-1][1] >= video_info["packets"]:
            raise ComposeError(f"the log shows src {events[-1][1]}, {args.video} has {video_info['packets']} frames")
        stages = stage_rows_before(run, summary) if args.label == "before" else stage_rows_after(rows, summary)
        for s in stages:
            log.info("stage %s [%s] = %s ms <- %s", s.key, s.device, fmt(s.value_ms, 3), s.origin)
            if s.value_ms is None:
                log.warning("stage %s: no value, shown as --", s.key)
        painter = Painter(args.label, stages, footer_lines(run, args.video, video_info))
        detections = DetectionReader(run / "detections.jsonl")
        stats = compose(args, rows, events, LiveValues(rows, cpu), painter, detections)
        if args.out:
            good = self_check(args, rows, events, summary, cpu, stats, detections, column)
            log.log(logging.INFO if good else logging.WARNING, "self-check %s", "passed" if good else "FOUND DIFFERENCES")
    except (ComposeError, OSError, subprocess.CalledProcessError, ValueError, KeyError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
