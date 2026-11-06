# 남은 응용 CPU 줄이기 — 변경 3건 전후 비교 (2026-11-06)

[`o2_perf_profile`](../o2_perf_profile/summary.md)의 perf 분해에서 나온 후보 중 세 건을 구현하고, 확정 구성 O2에서 변경 전 바이너리와 비교했다. 부모 실험은 [`technique_ablation`](../../summary.md)이다.

결정: 후처리 명령 버퍼 사전 기록, `--gpu-timestamps` 옵션(기본 on), HailoRT 사전 매핑 크기 맞춤을 구현한다 (사용자 결정, 2026-11-06)

## 변경

1. **HailoRT 사전 매핑 크기 맞춤** (`01_src/hailo/hailo_raw_runner.cpp`)
   - 사전 매핑(`dma_map_dmabuf`)의 크기를 dma-buf 전체 크기에서 텐서 크기(hw_frame_size)로 바꿨다. 이제 `infer_async`에 넘기는 크기와 같아 HailoRT 4.24 매핑 캐시에 적중한다.
   - strace로 센 hailo map ioctl은 300프레임에서 2147회 → 47회다. 줄어든 2100회는 프레임당 7회(입력 1 + 출력 6)와 맞는다.
2. **후처리 명령 버퍼 사전 기록** (`01_src/vulkan/yolo_postproc.cpp`)
   - 슬롯마다 한 번 기록해 다시 제출한다.
   - 푸시 상수 키(레터박스, 원본 크기, checksum)가 바뀔 때만 다시 기록한다.
3. **`--gpu-timestamps on|off`** (`01_src/pipeline/main.cpp`)
   - off면 전처리·후처리·NMS·표시 변환·오버레이의 타임스탬프 쓰기, 쿼리 리셋, 결과 읽기가 모두 빠진다.
   - GPU ms 열은 nan으로 나온다. CSV의 열 개수와 순서는 그대로다.
   - 기본값은 on이다(기본값 변경은 사용자가 따로 정한다).

**기능 동일성.** `--input-pace asap --frames 600 --display off`에서 세 바이너리의 `detections.jsonl`이 바이트까지 같다. md5는 `b453763e63d7b31f74109f1d75b37650`이다. 비교한 세 바이너리는 다음과 같다.
- 변경 전
- 변경 후 on
- 변경 후 off

나머지 검증 결과는 다음과 같다.
- ctest 221/221 통과.
- 검증 레이어 errors 0(O2·asap b8 × on/off).
- `--output-copy --input-copy` 300프레임 검출도 변경 전과 같다.
- NPU 원시 출력 6텐서도 바이트 일치한다(`yvo_hailo_raw_probe --probe runner`).

검증 로그는 `11_logs/2026-11-06_cpu_trim/`(git 밖)에 있다.

## 측정 조건

- 날짜: 2026-11-06(KST). 환경은 [`env_bench.txt`](env_bench.txt)에 있다.
- 하드웨어·소프트웨어: Raspberry Pi CM5, Hailo-8 PCIe Gen3 x1, V3D 1150 MHz, HailoRT 4.24.0, YOLO11m, `video_h265.mp4`(1816프레임).
- **CPU 거버너 performance**: 측정 동안만 바꾸고 ondemand로 되돌린 것을 확인했다.
- 구성은 `technique_ablation_bench.sh`의 `base`와 같다.
  - 배치 1/슬롯 1, 입력 페이싱 source, 입력 큐 1.
  - 표시 on(Vulkan 합성, wayland 전체 화면), 기록 on.
  - NHCW dma-buf 입력, GPU 후처리·NMS, HW 디코더 1스레드.
- 바이너리:
  - `base`: 변경 전, md5 `ef24741c7e5ca9457253f73e03aa73ca`. `dmabuf_ablation`·`o2_perf_profile`과 같은 바이너리다.
  - `new_*`: 변경 후, md5 `af63c101943f444a215e7eca0315b77b`. 커밋 전 작업 트리이며, 이 결과와 같은 커밋으로 들어간다.
- 반복·냉각: 3회 인터리브(회차가 바깥 루프)로 돌렸다. 매 실행 전 Hailo 칩을 65 °C 이하로 식혔다. perf는 켜지 않았다.
- 응용 CPU는 stdout의 `thread cpu yvo_pipeline … ms/processed` 값이다.

## 결과 (3회 중앙값, 괄호는 최소~최대)

| 설정 | 응용 CPU ms/처리 프레임 | 변경 전 대비 | fps | 입력 버림 | 도착→결과 평균 ms | 도착→화면 평균 ms |
|---|---:|---:|---:|---:|---:|---:|
| `base` (변경 전) | **1.448** (1.438~1.489) | — | 24.29 | 18.9% | 53.77 | 69.95 |
| `new_ts_on` (변경 1·2, 타임스탬프 on) | **1.348** (1.316~1.361) | −0.100 (−6.9%) | 24.31 | 18.9% | 53.80 | 70.27 |
| `new_ts_off` (변경 1·2·3, 타임스탬프 off) | **1.179** (1.178~1.215) | −0.269 (−18.6%) | 24.32 | 18.9% | 53.63 | 70.18 |

원자료는 [`results_raw.csv`](results_raw.csv)와 [`results.csv`](results.csv)에 있다. hailort.log의 warning·error는 9회 모두 0건이다.

## 결론

- **사전 매핑 크기 맞춤과 후처리 사전 기록이 함께 0.10 ms/처리 프레임(−6.9%)을 줄였다.**
  - perf 분해의 추정치(0.04 + 0.08, perf를 켠 척도)와 크기가 같은 쪽이다.
  - 두 변경을 따로 재지는 않았다.
- **타임스탬프를 끄면 0.17 ms가 더 준다.** V3DV에서 타임스탬프 쓰기와 쿼리 리셋은 CPU 잡이 되어 드라이버 제출 스레드의 비용이 된다.
  - 세 변경을 합치면 1.448 → 1.179 ms(−18.6%)다.
  - 대가는 GPU 단계별 ms 기록이 빠지는 것이다.
- 처리 FPS, 입력 버림, 도착→결과·도착→화면 지연은 실행 간 폭 안에서 같다.

## 주의·남은 일

- `base`의 이번 중앙값(1.448)은 `dmabuf_ablation`의 `base`(1.437)와 같은 바이너리, 같은 조건이다. 차이 0.011 ms는 실행 간 폭(약 0.05) 안이다.
- 문서 서술 갱신이 필요하다. "사전 매핑에 dma-buf 실제 크기를 쓴다"는 서술이 바뀐 계약과 어긋난다.
  - `08_learn/04_npu_hailort.md:576-582`
  - `08_learn/06_zerocopy_dmabuf.md:795`
  - `10_paper/paper_ko.md` §5.2
- `--gpu-timestamps`의 기본값(on 유지 또는 off)은 사용자 결정 대기다. GPU ms 열을 쓰는 벤치는 on을 명시해야 한다.
- 이 결과로 확정 구성의 헤드라인(응용 CPU 1.44 ms 등)이 바뀌는데, 논문 반영 여부는 정하지 않았다.
