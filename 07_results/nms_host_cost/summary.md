# 호스트 NMS 후처리의 CPU 비용과 원시 출력 전환 타당성

- 측정일: 2026-09-20
- 환경: Raspberry Pi CM5 (BCM2712), Hailo-8 **PCIe Gen3 x1**(`current_link_speed` = 8.0 GT/s로 확인), HailoRT 4.24.0
- 모델: `03_models/yolov11m.hef` (Multi Context 4개, 출력 `yolov8_nms_postprocess`)
- 목적: `07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md` §6의 미해결 항목
  "`libhailort` 0x30ffac 부근 함수를 확정한다 — 이것이 NMS인지 검증해야 한다"를 닫고,
  **원시 출력 HEF + GPU NMS** 설계의 타당성을 착수 전에 판정한다.

## 1. 0x30ffac은 NMS 후처리다

| 근거 | 내용 |
|---|---|
| HEF 사양 | 출력이 `yolov8_nms_postprocess`, `Op YOLOV8 Post-Process`, 80 클래스. HailoRT가 **호스트 CPU**에서 실행한다 (`evidence/hef_stream_infos.txt`) |
| gdb | HailoRT가 `std::thread`로 띄운 내부 워커 스레드에서 돈다. 진입 시 `x1 = 0x50 = 80`으로 **클래스 개수와 일치** (`evidence/gdb_backtrace.txt`) |
| 레이아웃 무관 | NHWC 3.60, NHCW 3.96 Mcycles/frame으로 거의 같다. 입력 경로를 바꿔도 안 변하는 **출력 쪽 작업**이다 |
| 우리 코드 아님 | `parse_nms_by_class` + letterbox 역변환은 프로파일 0.1% 아래로 잡히지 않는다 |

## 2. 비용은 검출 개수와 무관한 고정 비용이다

score threshold를 올려 검출을 2090개 → 0개로 없애도 CPU가 변하지 않았다 (`evidence/score_sweep.txt`).

| score | 검출 | CPU ms/frame |
|---|---:|---:|
| 0.2 | 2090 | 3.256 |
| 0.9 | 2 | 3.366 |
| 0.99 | 0 | 3.363 |

억제(suppression)가 아니라 **원시 텐서 전체를 훑는 작업**이 비용의 정체다.
640×640 YOLOv8 계열은 앵커 8,400개(80²+40²+20²) × (클래스 80 + DFL 64) ≈ 120만 개 값을
매 프레임 dequantize하며 스캔해야 하고, 이는 몇 개가 살아남든 동일하다.

## 3. 원시 출력은 PCIe 트래픽을 늘리지 않는다

`hailortcli parse-hef --parse-streams`가 보여주듯, 장치는 이미 **원시 텐서 6개를 매 프레임 내보내고 있다.**

| 스트림 | shape | 바이트 |
|---|---|---:|
| conv74 / conv71 | 80×80×80 / 80×80×64 | 512,000 / 409,600 |
| conv90 / conv87 | 40×40×80 / 40×40×64 | 128,000 / 102,400 |
| conv105 / conv102 | 20×**24**×80 / 20×**24**×64 | 38,400 / 30,720 |
| **합계** | | **1,221,120** |

`yolov8_nms_postprocess`는 **VStream 레벨의 호스트 연산**이다. 1.22 MB가 PCIe를 건너온 뒤
호스트가 그것을 160,320 B로 줄인다. 즉 **원시 출력으로 바꿔도 PCIe 트래픽은 늘지 않는다.**
(P5 텐서는 실제 W=20이 24로 패딩돼 있다. 소비하는 쪽이 열 20~23을 건너뛰어야 한다.)

## 4. 실측: full_async vs raw_async

`hailortcli run2 -t 15 --batch-size 8 --scheduling-algorithm round_robin`, 모드를 교대로 3회.
각 실행 전 칩 온도를 65 °C 이하까지 냉각했다.

| 모드 | FPS 중앙값 | FPS 범위 | CPU / 15 s | CPU ms/frame |
|---|---:|---|---:|---:|
| `full_async` (호스트 NMS 포함) | 51.22 | 51.21~51.22 | 2.78 s | 3.618 |
| `raw_async` (원시 6스트림) | **51.62** | 51.61~51.68 | **0.36 s** | **0.465** |

- **CPU 절감: 프레임당 3.15 ms**
- **FPS: +0.78%** — 두 모드의 범위가 겹치지 않는 재현되는 차이다. 원시 출력이 처리량을 **떨어뜨리지 않는다**는 §3의 예측이 확인됐다. 오히려 호스트 NMS가 아주 미세하게 파이프라인을 지연시키고 있었다.

**절감량 해석 주의.** 파이프라인 perf로 `0x30ffxx` 심볼만 집계하면 2.22 ms/frame인데, 이번 모드 차이는 3.15 ms/frame이다. 두 수치는 다른 것을 잰다:

- perf 2.22 ms = NMS 연산 심볼만
- 모드 차이 3.15 ms = NMS 연산 + 출력 변환(UINT8→FLOAT32) + 160 KB 출력 버퍼 처리

여기에 CPU 클럭 차이도 섞인다(두 실행 모두 코어 점유율이 낮아 거버너가 클럭을 내린다).
**"2.2~3.2 ms/frame"이 정직한 범위**이며, 어느 쪽이든 결론은 같다.

또한 `hailortcli`는 입력이 V3D dma-buf가 아닌 일반 호스트 메모리이고 디코드·GPU 단계가 없다.
**절대값이 아니라 두 모드의 차이만** 우리 파이프라인에 옮겨 적용해야 한다.

## 5. 결론

**원시 출력 + GPU NMS 설계는 유효하다.** 재컴파일이 필요 없고(원시 스트림이 이미 HEF에 있다),
PCIe 비용이 없으며, `raw_sync`/`raw_async` 모드가 HailoRT에 이미 존재한다.

| | 현재 | 원시 출력 + GPU NMS 예상 |
|---|---:|---:|
| FPS | 51.6 | 51.6~52 (불변 또는 미세 상승) |
| CPU | 3.8 ms/frame | 0.7~1.6 ms/frame |
| GPU | 2.13 ms/frame | 3~4 ms/frame (예산 19.3 ms) |

**다만 처리량은 오르지 않는다.** 병목은 PCIe/NPU이고, `hw_only`와 `streaming` FPS가 같다는
기존 벤치마크 결과가 이미 호스트 NMS가 처리량을 제한하지 않음을 보여줬다.
사는 것은 **CPU 코어 약 0.2개**다. 다중 스트림이나 트래킹 같은 후단 처리를 올릴 계획이 있다면
타당한 투자이고, 파이프라인을 더 빠르게 만드는 것이 목적이라면 병목(PCIe/NPU)을 건드려야 한다.

GPU 적합성은 좋다. 출력이 `NHCW`라 한 앵커의 클래스 값들이 W 간격으로 떨어져 있는데,
invocation을 x축에 매핑하면 인접 invocation이 연속 주소를 읽어 **완벽히 coalesce된다.**
CPU를 괴롭힌 레이아웃이 GPU에는 유리하다.

## 6. 남은 일

- `InferModel`이 postprocess를 건너뛸 수 있는지 확인한다. 불가하면 `ConfiguredNetworkGroup`의
  raw stream API로 러너를 상당 부분 다시 써야 하며, 이것이 구현 난이도를 가른다.
- 출력 텐서별 quantization scale/zero-point를 HEF에서 읽어 셰이더에 넘기는 경로가 필요하다.
- NPU 출력을 Vulkan이 할당·export한 dma-buf에 직접 받는 구성(입력 경로의 대칭형).
- NMS 억제 단계는 GPU에서 까다롭다. 검출이 프레임당 5.21개뿐이므로
  **GPU 디코드·압축 + CPU 억제 하이브리드**가 현실적이다. 압축된 수 KB를 순차로 읽는 것은
  기존에 문제가 된 "stride 3으로 1.2 MB 읽기"와 전혀 다르다.

## 7. 파일

- `results.csv`(모드별 중앙값), `results_raw.csv`(실행별), `env_bench.txt`
- `raw/{full_async,raw_async}_run{1,2,3}.log` — hailortcli 출력
- `evidence/hef_stream_infos.txt` — 원시 스트림 6개 사양
- `evidence/gdb_backtrace.txt` — 0x30ffac이 HailoRT 워커 스레드임을 확인
- `evidence/score_sweep.txt` — 고정 비용임을 보이는 threshold 스윕
- `evidence/perf_{nhwc,nhcw}_1816_top.txt` — 전체 1816프레임 프로파일 (시작 비용이 희석된 값)
