# 앱(libavcodec) 대 GStreamer 연구 경로: HW 디코드 → Vulkan SAND 전처리 → Hailo

## 측정 조건
- 측정 일시: 2026-09-29 06:55~07:41 KST
- 목적: 같은 HW HEVC 디코더 출력을 두 방식으로 받아 비교한다.
  - **앱 `yvo_pipeline`**: libavcodec DRM_PRIME으로 받는다.
  - **GStreamer 연구 경로 `yvo_gst_pipeline`**: `v4l2slh265dec` DMABuf를 자체 요소 `yvosandletterbox` → `yvohailosink`로 받는다.
  - 두 경로 모두 Vulkan 전처리(`SandLetterbox`, NHCW)와 Hailo 추론(`HailoRunner`, host NMS)은 **같은 코드**를 쓴다. 다른 것은 디코드와 프레임 전달 방식뿐이다.
  - GStreamer 경로를 만든 계기는 gst-plugins-bad rpt4로 `v4l2slh265dec`의 DMABuf 출력이 복사 없는 경로가 된 것이다([`decode_benchmark/runs/gst_rpt4`](../decode_benchmark/runs/gst_rpt4/summary.md)). 설계는 `01_src/gstelem/`에 있다.
- 환경(자세한 값은 [`env_bench.txt`](env_bench.txt))
  - 하드웨어·펌웨어: Raspberry Pi 5, 부트로더 2026-09-25, `v3d_freq=1150`, PCIe **Gen3** x1
  - 소프트웨어: kernel 6.18.50, gst-plugins-bad `1.26.2-3+rpt4+deb13u3`, HailoRT 4.24.0, git `bc76a85`
- 모델·입력: `03_models/yolov11m.hef`, `04_data/videos/video_h265.mp4`(1816프레임 전체)
- 명령: `RUNS=3 DETS=1 bash 06_tools/bench/gst_pipeline_bench.sh`
  - 설정 6개(`id|tool|batch|slots`): `app_nhcw_b1|app|1|2`, `gst_b1|gst|1|2`, `app_nhcw_b4|app|4|8`, `gst_b4|gst|4|8`, `app_nhcw_b8|app|8|16`, `gst_b8|gst|8|16`
  - 3회를 run 바깥, 설정 안쪽으로 인터리브했다. 실행 전마다 칩을 65 °C 이하로 식혔다.
  - app 인자는 `--mode dmabuf --layout nhcw --batch B --slots S`(`pipeline_bench.sh`의 `nhcw_*`와 같음), gst 인자는 `--sink hailo --batch B --slots S`다.
  - `DETS=1`로 측정이 끝난 뒤 설정마다 1회 더 돌려 검출을 비교했다. 이 실행은 시간 측정에서 뺐다.
- 신뢰성
  - 36회 모두 exit 0이고, `hailort.log` warning 0, error 0, 불안정 판정 0회다.
  - `throttled=0x80000`은 부팅 이후 soft temp limit가 한 번 있었다는 과거 기록 비트다. 현재 비트(하위 4비트)는 0이다.

## 결과 (중앙값, 3회)

| 설정 | fps | fps 범위 | cpu_ms_per_frame | 실행별 CPU (3회) | gpu_ms_mean | npu_ms_mean | e2e_ms_mean ¹ |
|---|---|---|---|---|---|---|---|
| app_nhcw_b1 | 24.420 | 24.42~24.42 | 3.358 | 3.446 / 3.358 / 3.345 | 1.817 | 79.86 | 122.52 |
| **gst_b1** | 24.430 | 24.43~24.43 | **3.142** | 3.095 / 3.142 / 3.202 | 1.760 | 79.88 | 120.78 |
| app_nhcw_b4 | 44.510 | 44.50~44.52 | 3.266 | 3.324 / 3.266 / 3.170 | 1.981 | 176.93 | 201.27 |
| **gst_b4** | 44.550 | 44.55~44.55 | **3.060** | 2.871 / 3.060 / 3.060 | 1.921 | 177.03 | 199.40 |
| app_nhcw_b8 | 51.640 | 51.64~51.65 | 3.254 | 3.254 / 3.560 / 3.225 | 1.978 | 306.13 | 327.28 |
| **gst_b8** | 51.690 | 51.69~51.69 | **3.050** | 3.050 / 3.072 / 2.859 | 1.924 | 306.25 | 325.42 |

¹ e2e 시작점이 다르다. 앱은 디코드가 프레임을 돌려준 시각, gst는 레터박스 요소에 프레임이 도착한 시각이다.

다른 열(p90, 온도, 냉각 시간)은 [`results.csv`](results.csv)에, 실행별 값은 [`results_raw.csv`](results_raw.csv)에 있다. gst 전용 진단(steady FPS, 슬롯 대기 등)은 [`results_raw_gst.csv`](results_raw_gst.csv)에 있다. gst의 `decode_ms_mean`은 해당 값이 없어 `nan`이다.

검출 비교(측정 후 1회, [`evidence/compare_b*.txt`](evidence/)):

| batch | 앱 검출 | gst 검출 | 매칭 @ IoU 0.5 / 0.7 / 0.9 | 좌표 오차 | `detections.jsonl` |
|---|---|---|---|---|---|
| 1 | 9457 | 9457 | 1.0000 / 1.0000 / 1.0000 | 0 px | **바이트 동일** |
| 4 | 9457 | 9457 | 1.0000 / 1.0000 / 1.0000 | 0 px | **바이트 동일** |
| 8 | 9457 | 9457 | 1.0000 / 1.0000 / 1.0000 | 0 px | **바이트 동일** |

## 결론
- **GStreamer 경로는 앱과 같은 결과를 같은 처리량으로 낸다.**
  - 검출은 세 배치 모두 `detections.jsonl`까지 바이트 단위로 같다.
  - FPS 차이는 +0.04~0.10%로, 둘 다 NPU 상한(Gen3 b8 51.7)에 붙어 있다. 차이는 측정 구간 정의(아래 주의) 수준이다.
  - NPU 지연(79.9 / 177.0 / 306.2 ms)도 같다. 같은 slots를 주면 NPU 대기열 깊이가 같아진다.
- **CPU는 GStreamer 경로가 설정마다 약 0.2 ms/frame(6.3~6.4%) 낮다.**
  - b1 3.36 → 3.14, b4 3.27 → 3.06, b8 3.25 → 3.05 ms/f다.
  - 세 배치 모두 실행별 범위가 겹치지 않는다(app 3.17~3.56, gst 2.86~3.20).
  - 측정 구간의 차이로는 설명되지 않는다. gst는 preroll 동안 링 슬롯만큼(b8에서 약 20프레임) 디코드·전처리를 구간 밖에서 끝내지만, 그 몫은 최대 약 1%(≈0.035 ms/f)다.
  - 원인은 가르지 못했다. 디코드·역다중화 쪽 호스트 비용(libavcodec의 demux·parse·DRM_PRIME 프레임 관리 대 qtdemux·h265parse·v4l2codecs)의 차이로 추정한다(**추정**, perf로 확인하지 않았다).
- **GPU 전처리 시간도 같은 수준이다**(gst 1.76~1.92 ms 대 앱 1.82~1.98 ms, 셰이더와 입력이 같다). 약 3%의 차이는 판정하지 않는다.
- 따라서 "앱 본체가 libavcodec을 쓰는 이유"는 이제 성능이 아니다. rpt4 이후 GStreamer 경로는 **같은 정확도, 같은 처리량, CPU는 오히려 조금 낮다.**
  - 대신 GStreamer 경로는 패키지 버전에 민감하다. rpt3에서는 같은 디코더가 32.7 FPS / 30.6 ms/f였다.
  - 요소가 DMA_DRM이 아닌 caps를 거부하므로 복사 경로로 조용히 빠지지는 않는다. 그래도 배포 환경에 rpt4 이상이 필요하다.

## 주의
- **측정 구간 정의가 다르다.**
  - gst의 wall과 CPU는 PAUSED preroll(요소 초기화, 링 채우기) 뒤 PLAYING부터 EOS까지다. 앱은 초기화 뒤 첫 디코드부터 drain까지다.
  - gst의 steady FPS(첫 sink 버퍼부터 마지막까지)는 b8 52.09로 PLAYING→EOS 값(51.69)보다 높다. FPS 비교에는 두 도구가 같은 방식으로 내는 `fps`만 쓴다.
- e2e는 시작점이 달라 직접 비교하지 않는다(위 ¹). 앱 쪽 시작점이 q_dec 대기(최대 2프레임)만큼 앞선다.
- CPU 차이의 원인은 perf로 확인하지 않았다. 확정하려면 두 도구를 `perf record`로 같은 구간 프로파일해 디코드 쪽 심볼을 비교해야 한다.
- 연구 경로는 flush·seek를 지원하지 않는다. 입력 영상은 1816프레임짜리 하나만 확인했다(gst는 `--expect-frames 1816`으로 검사).
- `samples/`에는 검출 비교용 `detections.jsonl`(앱과 gst 사본이 바이트 동일)과 앱의 첫 프레임 PNG가 있다.
