---
name: measure
description: >
  이 프로젝트의 성능 측정 프로토콜. NPU·GPU·디코드·전체 파이프라인의 처리량(FPS)·지연·CPU 비용을
  재거나, 07_results/ 아래에 측정 결과를 기록할 때 쓴다. 냉각 규칙, 측정 전 환경 확인,
  반복·중앙값, hailort.log 경고로 판정하는 신뢰성 기준, 결과 폴더·CSV·summary.md 규약을 담는다.
  트리거: "측정", "벤치마크", "FPS 재", "성능 재", "지연 측정", "latency", "throughput",
  "얼마나 빠른지", "benchmark", "07_results에 기록".
---

# 측정 프로토콜

성능 수치는 조건을 적지 않으면 쓸모가 없고, 과열 상태에서 재면 **틀린 값이 조용히 기록된다**.
이 프로젝트에는 이미 그렇게 버려진 측정이 있다(`07_results/npu_benchmark/runs/gen3/summary.md`의
"신뢰 불가" 표). 아래 순서를 지킨다.

## 0. 기존 스크립트를 먼저 본다

새로 짜기 전에 확인한다. 대개 `configs` 배열에 행 하나를 더하는 것으로 끝난다.

`06_tools/bench/` 에 **8개**가 있다. 목록·환경변수는 [`06_tools/README.md`](../../../06_tools/README.md).

| 스크립트 | 대상 | 출력 |
|---|---|---|
| `pipeline_bench.sh` | 전체 파이프라인. `{mode} × {layout} × {batch/slots} × RUNS` | `07_results/pipeline_ffmpeg_vulkan_hailo/` |
| `op_point_bench.sh` | 운용점 스윕(batch·slots·gpu_depth·postproc·display) | `07_results/op_point/` |
| `display_bench.sh` | 표시 통합 e2e (RESULT_CSV + RESULT_CSV2 병합, 59열 — 48·53열 옛 CSV 와 섞지 않는다. `kms` 백엔드는 첫 커밋이 블로킹 모드셋이라 그 직후 프레임을 집계에서 뺀다) | `07_results/display_e2e/` |
| `decode_bench.sh` | 디코드 경로 비교 | `07_results/decode_benchmark/` |
| `backpressure_bench.sh` | 백프레셔 전후 비교 | `07_results/raw_runner_backpressure/` |
| `isp_bench.sh` | ISP 전처리 단독(NPU 미사용) | `07_results/isp_resize/` |
| `isp_pipeline_bench.sh` | ISP 포함 전체 파이프라인 | `07_results/isp_resize/runs/pipeline/` |
| `gst_pipeline_bench.sh` | 앱(`yvo_pipeline` dmabuf/NHCW) 대 GStreamer 연구 경로(`yvo_gst_pipeline --sink hailo`), b1/b4/b8 같은 slots 인터리브. CSV 앞 26열은 `pipeline_bench.sh` 와 같다(gst 의 `decode_ms` 는 nan) | `07_results/gst_pipeline/` |

> ⚠ `pipeline_bench.sh` 의 `configs` 스키마(`id|mode|layout|batch|slots` 5필드)는 **바꾸지 마라.**
> 다른 스크립트들이 대조군 기준으로 쓴다.
>
> 여덟 스크립트 모두 시작할 때 `build/.bench_running` 에 PID 를 쓰고 `trap` 으로 지운다.
> `npu_guard` 훅이 이걸 보고 측정 중 빌드·테스트를 막는다. **락을 우회하지 마라** —
> 측정 중 다른 작업이 돌면 CPU 시간이 에러 없이 오염된다.

`pipeline_bench.sh`의 유용한 환경변수:
`RUNS`, `FRAMES`, `ONLY`(설정 id의 grep 패턴), `APPEND=1`(기존 CSV에 이어붙이기),
`COOL_C`/`COOL_MAX_S`/`HOT_C`, `OUT_DIR`, `SAMPLES`, `SAMPLE_LAYOUT`.

```bash
RUNS=1 FRAMES=300 bash 06_tools/bench/pipeline_bench.sh    # 짧게 확인
ONLY='nhcw' APPEND=1 bash 06_tools/bench/pipeline_bench.sh # 일부만 재서 기존 결과에 추가
```

## 1. 측정 전 환경 확인 — 기록까지 한다

**추정하지 말고 확인한다.** PCIe 세대는 NPU 처리량을 거의 배로 바꾸는 가장 큰 변수다.
과거에 에이전트 정의에 적혀 있던 낡은 값을 그대로 인용해, Gen3에서 잰 측정을 "Gen2"로 보고한 적이 있다.

```bash
# PCIe 링크 — 권한 없이 읽히는 권위 있는 출처다. lspci -vv 의 LnkSta: 는 root 가 아니면
# 아무것도 출력하지 않으므로 쓰지 않는다.
cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed
cat /sys/bus/pci/devices/0001:01:00.0/current_link_width
vcgencmd measure_temp; vcgencmd get_throttled; vcgencmd measure_clock arm
uname -r
```

| current_link_speed | 세대 |
|---|---|
| `2.5 GT/s PCIe` | Gen1 |
| `5.0 GT/s PCIe` | Gen2 |
| `8.0 GT/s PCIe` | Gen3 |
| `16.0 GT/s PCIe` | Gen4 |

두 벤치 스크립트는 `pcie_link()` 함수로 이 줄을 `env_bench.txt`에 자동으로 남긴다.
링크 세대는 `/boot/firmware/config.txt`의 `dtparam=pciex1_gen`으로 바뀌므로, **이 문서를 포함해
어떤 문서에 적힌 값도 믿지 말고 읽는다.**

Hailo 칩 온도는 `hailo_platform`으로 읽는다(`pipeline_bench.sh`의 `hailo_temp()`를 그대로 쓸 것):

```bash
/home/pi/miniconda3/envs/hailo-yolo/bin/python -c \
 "from hailo_platform import Device; d=Device(); t=d.control.get_chip_temperature(); \
  print(f'{max(t.ts0_temperature,t.ts1_temperature):.1f}'); d.release()"
```

이 스냅샷을 `07_results/<실험>/env_bench.txt`에 남긴다.

## 2. 냉각 규칙

Hailo 칩은 연속 부하에서 90 °C대까지 오르고, 그러면 클럭이 400 → 250 MHz로 떨어진다.

- 실행 직전 칩 온도가 **75 °C를 넘으면** 65 °C 이하가 될 때까지 기다린다(최대 120초).
- **직전 실행의 `hailort.log`에 온도·클럭 경고가 있으면 무조건 냉각한다.**
- 냉각 대기 시간도 기록한다. 측정 전체 소요의 절반 이상일 수 있다.

## 3. 반복과 집계

- 설정당 **3회**, **중앙값**을 보고하고 **최소·최대(또는 편차)** 를 함께 적는다.
- 반복(run)을 바깥 루프, 설정을 안쪽 루프에 둔다. 열 상태 편향을 설정 간에 고르게 나눈다.

## 4. 신뢰성 판정 — 이게 핵심이다

실행마다 `hailort.log`를 스캔한다:

```bash
grep -E '\[warning\]' hailort.log | grep -cE '[Tt]emperature|[Tt]hrottl|[Oo]verheat|clock'
```

**0이 아니면 그 실행의 성능 수치는 신뢰할 수 없다.** 다음 중 하나를 한다:

1. 칩을 식히고 재측정한다(권장).
2. 재측정이 당장 어려우면 요약표에 **"신뢰 불가"로 명시**하고, 왜 그런지(클럭 저하 등) 적는다.

신뢰 불가 값을 아무 표시 없이 요약표에 싣지 않는다. 나중에 그 값이 인용된다.

## 5. 결과 저장 규약 — `07_results/<실험>/`

### 무엇이 "실험"인가

**`07_results/README.md` 의 실험 목록표가 정본이다.** 거기 한 행으로 오르는 것이 실험이고,
그 아래 하위 폴더는 **같은 실험의 측정 회차·변종**이다(새 실험이 아니다).

예: `npu_benchmark/runs/{gen2,gen3}` 는 같은 벤치의 PCIe 세대별 회차,
`op_point/runs/latency_sweep/` 은 운용점 실험의 지연 스윕 회차,
`isp_resize/runs/{pipeline,pipeline_rerun}/` 은 ISP 실험의 파이프라인 단계 1·2차다.
**회차 폴더가 자기 `summary.md` 를 가질 수 있다** — 그렇다고 별도 실험으로 승격하지 않는다.
새 실험으로 올릴지 헷갈리면 기준은 하나다: **README 표에 한 행으로 설명되는가.**

### 레이아웃 — 정본은 `STRUCTURE.md` §3

폴더 레이아웃·경계·이동 원칙의 정본은 [`STRUCTURE.md`](../../../STRUCTURE.md) §3 이다(기계 검사: `python3 06_tools/python/check_structure.py`).
측정할 때 필요한 요점만:

- 공통 필수: `summary.md`, `env_bench.txt`(§1 의 환경 스냅샷). **스윕형**은 `results.csv`·`results_raw.csv`·`raw/` 도 필수,
  **프로브형**은 `evidence/` 가 필수. 유형은 `07_results/README.md` 색인 표의 "유형" 열에 적는다.
- 실행 1회 = `raw/<설정>_run<N>/` 폴더 하나(`stdout.log` 전문 또는 `run.log` 발췌 + `hailort.log`).
  벤치의 온도 프로브가 남기는 `hailort.log` 는 `raw/_session/` 에 생긴다. `raw/` 바로 아래에 파일을 두지 않는다.
- 같은 실험의 회차·변종은 `runs/<회차>/`(깊이 1). 1회 덤프(PNG, `detections.jsonl`)는 `samples/`.
- **`evidence/` 와 `verification/` 은 목적으로 나눈다.** 결론의 근거(관문 통과·원인 특정·정확도 판정)는 `evidence/`,
  변경이 아무것도 깨지 않았다는 게이트 로그(verify 스킬 단계)는 `verification/`. 도구가 perf 든 hailortcli 든 상관없다.
- **보관된 실험 폴더는 덮어쓰지 않는다.** 벤치 스크립트는 `OUT_DIR` 이 이미 있고 비어 있지 않으면 멈춘다(`guard_out_dir`).
  같은 폴더에 이어 쓰거나(`APPEND=1`) 다시 재야 하면 **`FORCE=1`** 을 명시한다. C++ 도구의 기본 출력은 `build/out/<도구>/` 이고,
  `07_results` 에 쓸 때는 `--out-dir` 을 준다.
- 결과 파일의 **내용**은 고치지 않는다. 없는 파일을 나중에 만들지 않는다(재구성은 날조) — 그래서
  `display_vulkan_hdmi`·`gpu_postprocess`·`npu_benchmark` 의 `env_bench.txt` 누락은 규칙표 예외로 둔다.
  위치만 바꾸는 이동은 `relocate.py` 와 `STRUCTURE.md` §6 매핑표로 한다.

**중복 산출물을 지울 때의 순서** — `07_results/` 안의 PNG 중 상당수는 바이트가 같은데,
**그 동일성 자체가 결론**인 경우가 있다(host 경로 == GPU 경로, A안 == C안,
batch 4 == batch 8, NHWC == NHCW). 그래서 지우되 근거는 남긴다:

1. `md5sum` 으로 **바이트 일치를 확인**한다. 하나라도 다르면 건드리지 않는다
2. md5·남긴 파일·지운 파일을 [`07_results/duplicate_png.md`](../../../07_results/duplicate_png.md) 에 기록한다
3. **문서에 인용된 파일은 지우지 않는다**(`python3 06_tools/python/check_links.py` 로 확인)
4. 수치의 1차 근거(`detections.jsonl`, CSV)는 **정본 1부를 반드시 남긴다.** 바이트 일치 사본만 1~3 순서로 지울 수 있다

**CSV는 LF 줄바꿈으로 쓴다.** 파이썬 `csv.writer`는 기본이 CRLF다. 반드시 지정할 것:

```python
csv.writer(f, lineterminator="\n")
```

(이걸 놓쳐서 벤치 스크립트의 CSV 헤더 검사가 실패한 적이 있다.)

## 6. `summary.md` 형식

순서: **측정 조건 → 표 → 결론 → 주의·남은 일**

- 조건: 날짜, 하드웨어, 링크 속도, 모델, 입력 영상, HailoRT/드라이버 버전, 반복 횟수
- 표: 설정별 중앙값. 단위를 열 이름에 넣는다(`fps`, `cpu_ms_per_frame`, `e2e_ms_mean`)
- 결론: 병목이 어디인지, 무엇이 한계인지를 문장으로
- 주의: 신뢰 불가 값, 온도 상황, 측정하지 않은 설정

상세 보고가 필요하면 `REPORT.md`를 따로 쓰고 `summary.md`에서 링크한다.
기존 예: `07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md`.

## 알아 둘 기준선

아래는 **과거 측정 기록**이다. 현재 상태가 아니다. 조건이 다르면 직접 비교하지 않는다.

| 항목 | 값 | 측정 조건 | 출처 |
|---|---|---|---|
| NPU 상한 (`hailortcli benchmark`) | batch 8에서 **51.7 FPS** | 2026-09-20, PCIe **Gen3** x1 | `07_results/npu_benchmark/runs/gen3/` |
| 같은 항목, 한 세대 아래 | batch 8에서 **26.1 FPS** | 2026-09-20, PCIe **Gen2** x1 | `07_results/npu_benchmark/runs/gen2/` |
| HW HEVC 디코드만 (libavcodec DRM_PRIME) | **398.5 FPS**, CPU 0.23~0.28 ms/frame | 1080p30 | `07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md` §4.1 (**`decode_benchmark/` 아님**) |
| GPU 전처리 셰이더 (1080p → 640×640) | 2.13 ms (연속 스트림) | V3D 7.1, local 32×8 | `07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md` |
| 전체 파이프라인 (dmabuf, NHCW, b8/s16) | 51.6 FPS, CPU 약 3.5 ms/frame | 2026-09-20, PCIe **Gen3** x1 | 같은 폴더 `results.csv` |

Gen2와 Gen3가 두 배 가까이 차이 난다. **기준선을 인용하기 전에 지금 세대를 읽는다.**

GPU를 단발로 실행하면 DVFS 때문에 셰이더 시간이 부풀려진다. 연속 스트림에서 잰 값을 쓴다.
