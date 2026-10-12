# 측정 결과 색인

각 폴더는 `.claude/skills/measure` 규약을 따른다(설정당 3회·중앙값, 냉각 게이트,
`hailort.log` 경고로 신뢰성 판정, `summary.md` 필수).
**수치를 인용하기 전에 해당 `summary.md` 의 "주의·남은 일" 절을 먼저 읽어라.**

---

## 지금 권고하는 파이프라인

```
디코드(HW HEVC, DRM_PRIME dma-buf)
  → GPU 전처리 sand_letterbox.comp (NHCW 직접 출력, zero-copy)
  → NPU 원시 출력 6텐서 (infer_async, dma-buf H2D/D2H)
  → GPU 후처리 yolo_decode.comp + efficient_nms.comp (경계 프레임만 CPU 억제 fallback)
  → GPU 오버레이 overlay_boxes.comp + display_convert.comp
  → HDMI (Wayland 스왑체인)
```

> GPU NMS(`efficient_nms.comp`, 2026-09-22)와 HVS 평면 표시 백엔드(`--display-backend wayland-planes` / `kms`)는
> 2026-10-08 정식 측정을 마쳤다 — [`gpu_nms_e2e/`](gpu_nms_e2e/summary.md)(회귀 없음·검출 바이트 동일),
> [`display_planes_e2e/`](display_planes_e2e/summary.md)(평면 경로 CPU −0.44 ms/f·V3D −72%, 지연은 같음, 슬롯 6 필요). 아래 수치는 모두 그 이전 구성의 값이다.
>
> **측정 조건(2026-09-20 전후): V3D 960 MHz · FFmpeg 디코더 자동 5스레드 · CPU NMS.** 2026-09-29 에 둘이 바뀌었다.
> - 부트로더 2026-09-25 로 V3D 클럭이 **1150 MHz** 가 됐다. FPS·CPU 는 불변, GPU 전처리만 약 14% 빨라졌다([`bootloader_20260925`](pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md)).
> - 앱 디코더 프레임 스레드 기본값이 **1** 이 됐다(5032575). CPU·e2e 가 줄고 FPS·검출은 같다([`dec_threads/`](dec_threads/summary.md)).
> - 그 회차의 GPU 후처리·표시 설정은 기본값 `--nms gpu` 로 돌아 **GPU NMS 의 비정식 값**이 있다(`gpu_nms_mean` 0.036~0.050 ms, p90 약 0.057, 폴백 0).
>   정식 A/B 는 아니다.

```bash
XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 \
build/bin/yvo_pipeline \
  --video 04_data/videos/video_h265.mp4 --hef 03_models/yolov11m.hef \
  --mode dmabuf --layout nhcw --postproc gpu \
  --batch 2 --slots 4 \
  --display on --display-fullscreen --display-sync matched --display-pace source
```

위 커맨드(`--display-pace source`)의 실측:
**29.91 FPS(소스 30 fps 추종) · glass-to-glass median 101.20 ms(p90 118.36) · CPU 2.18 ms/frame ·
프레임 드롭 0 · 검출 3369(변화 없음)**.
`--display-pace asap` 으로 바꾸면 **34.87 FPS · glass 130.75 ms · CPU 1.84 ms/frame** 이다.
근거는 [`display_e2e/summary.md`](display_e2e/summary.md) §3·§3.2(운용점 확정은 [`op_point/summary.md`](op_point/summary.md) §8).
⚠ 두 페이싱은 **다른 실행**이다. FPS·CPU·glass 를 섞어 인용하지 마라.
현재 기본값(1스레드·1150 MHz·GPU NMS)의 `source` 참고치는 CPU **1.78 ms/frame**(같은 회차 자동 스레드 2.07), glass **평균** 99.5 ms,
드롭 0, 검출 3369 다([`dec_threads/`](dec_threads/summary.md)). 위 101.20 은 **중앙값**이다 — 섞지 않는다. `asap` 은 다시 재지 않았다.

운용점은 용도에 따라 고른다 — 최대 처리량은 `b8/s16`(51.5 FPS, 지연 325 ms — 자동 5스레드 조건, 1스레드 기본값에서는 318.6 ms,
[`dec_threads/`](dec_threads/summary.md)), **최소 지연(O2)은 `--batch 1 --slots 1 --input-pace source --input-queue 1`** 이다(표시 off 기준값, 화면까지는 Vulkan 합성 표시로 도착 → 화면 평균 73.1 ms).
프레임이 30 fps 시각에 도착하게 하고 넘치는 프레임은 가장 오래된 것부터 버린다. **도착 기준 평균 58.3 · p90 72.7 · max 75.3 ms**,
처리 24.41 FPS 라 **입력의 18.6% 를 버린다**(검출은 처리한 프레임만 — 900 도착에 2750) ([`input_pacing/`](input_pacing/summary.md), 현재 기본값 조건).
- 결정: 최소 지연 운용점(O2) = `b1/s1 + --input-pace source --input-queue 1` (사용자 결정, 2026-10-12).
  옛 O2 `b1/s1`(페이싱 없음, 디코드 끝 기준 81.6 ms — 자동 5스레드·960 MHz, 지금 기본값 79.19 ms)는 그중 약 42 ms 가 파일을 앞서 읽어 생긴 슬롯 대기였다
  ([`latency_b1s1_20261012`](op_point/runs/latency_b1s1_20261012/summary.md)). `op_point/summary.md` §5.5·§8 의 O2 표기는 그 시점 기록이다.
- 결정: 표시 on 의 최소 지연 운용점 = O2 + Vulkan 합성 표시(`--display on --display-fullscreen --display-sync matched`) — 도착 → 화면 평균 73.1 · p90 82.6 ms, 버림 22.2% (사용자 결정, 2026-10-12, [`display_on`](input_pacing/runs/display_on/summary.md)).
- 지연의 기점을 섞지 않는다 — **도착**(`e2e_arr`, 페이싱), **도착 → 화면**(`a2p`, 페이싱 + 표시), **디코드 끝**(e2e, 페이싱 없음), **슬롯 획득**(glass-to-glass, 표시). 섞어 인용하지 않는다.

---

## 이 하드웨어에서 설계를 지배하는 사실 (전부 실측)

1. **zero-copy 는 `dmabuf + NHCW` 에서만 성립한다.** V3D GEM mmap 포인터는 Hailo 가 DMA 매핑하지 못한다.
2. **GPU 가 쓴 버퍼를 CPU 로 훑으면 약 10배 비싸다.** V3D 메모리 타입이 `HOST_COHERENT` 하나뿐이고
   `HOST_CACHED` 가 없다. **반대 방향(CPU→GPU 쓰기)은 싸다.**

   ⚠ **순차 읽기 대역폭은 서로 다른 두 측정이 있다. 섞어 인용하지 마라.**

   | 측정 | 조건 | 결과 | 정본 |
   |---|---|---|---|
   | **P2 프로브** (할당처 비교) | 1,221,120 B, 30회 반복 + 워밍업 3회 | 0.826 ms = **1.479 GB/s**<br>(malloc 0.244 = 5.001, 캐시 dma-heap 0.396 = 3.084) | [`gpu_postprocess/summary.md`](gpu_postprocess/summary.md) P2 ⑥ |
   | **`yvo_preproc_bench`** (파이프라인 버퍼) | 1,228,800 B | 0.85~0.87 ms = **약 1.41 GB/s** | [`pipeline_ffmpeg_vulkan_hailo/summary.md`](pipeline_ffmpeg_vulkan_hailo/summary.md) §주의 |

   **할당처 간 비교에는 P2(1.479)** 를 쓴다 — 반복 횟수가 많고 malloc·dma-heap 대조군이 있다.
   stride-3 읽기는 같은 양에 약 14 ms = **0.088 GB/s** 이고, 16~17배 차이가 이 설계의 근거다.

   > 2026-09-21 정정: 이 색인이 한때 1.48(= P2 반올림)로 적었다가 1.41(= preproc_bench)로 "고쳤는데",
   > **둘은 애초에 다른 측정이었다.** 하나로 합치려 한 것이 실수였다.
3. **처리량 상한은 PCIe Gen3 x1 이 정한다.** `T(B) = 24.8 + 16.2·B ms`, 점근 상한 61.6 FPS, 실측 51.7.
4. **파이프라인에서 지연은 slots 가, 처리량의 천장은 batch 가 정한다.** 지연 ≈ `slots × 프레임 주기`, FPS = min(`B / T(B)`, 슬롯 제약)이고 천장은 slots = 2B 에서 닿는다([`grid_gpu_postproc`](op_point/runs/grid_gpu_postproc/summary.md), 2026-10-08 — `op_point/summary.md` §4 의 "batch 무의미"·§4.3 을 정정).
5. **표시 해상도를 낮추면 오히려 비싸진다.**(V3D 960 MHz 측정) 1:1 경로 **2.04 ns/px** vs 스케일 경로 **5.72**(720p) / **5.31**(900p) ns/px.
   1920×1080 1:1 이 기본. (픽셀당 값은 `시간 ÷ 픽셀 수` 로 계산한 것이다 — 원문은 ms 만 준다.)
6. **원시 출력에 `InferModel` 은 쓸 수 없다** — `VDevice::configure()` + `infer_async()` 저수준 경로 필수.

---

## 실험 목록 (실험 등록부)

**이 표가 "무엇이 실험인가"의 정본이다.** 한 행 = 실험 하나 = `07_results/` 아래 폴더 하나.
`check_structure.py` 가 이 표를 읽어 폴더 ↔ 행 1:1 과 **유형**별 필수 항목(스윕: `results.csv`·`results_raw.csv`·`raw/`, 프로브: `evidence/`)을 검사한다.
**상태**의 인용 제한은 아래 "⛔ 인용 금지 구간" 이 정본이다. 행 순서는 파이프라인 단계 순, 그다음 2026-09-20 통합 작업이다.

| 폴더 | 유형 | 상태 | 단계 | 회차 | 무엇을 쟀나 | 핵심 결론 |
|---|---|---|---|---|---|---|
| [`npu_benchmark/`](npu_benchmark/summary.md) | 레거시 | 부분 대체 | 1 NPU | `runs/{gen2,gen3,gen3_cooled}` | NPU 단독 처리량·지연 (Gen2 vs Gen3, batch 1~32) | batch 8 에서 **51.71 FPS**. Gen2→Gen3 로 두 항이 정확히 반씩 줄어 **PCIe 가 병목**임이 확정. ⚠ overall latency 표는 [`op_point/runs/latency_gen3_cooled/`](op_point/runs/latency_gen3_cooled/summary.md) 가 대체 |
| [`decode_benchmark/`](decode_benchmark/summary.md) | 스윕 | 유효 | 2 디코드 | — | 디코드 경로 비교 (SW/HW × H.264/H.265) | GStreamer `v4l2slh265dec` 는 HW 를 쓰고도 32.7 FPS(CPU 30.6 ms/f, ORC 복사, **rpt3 기준**). 같은 HW 를 ffmpeg CLI `-hwaccel drm` 로 쓰면 **332.8 FPS / CPU 2.0 ms/f**(CPU 내려받기 포함). 회차 [`runs/gst_rpt4`](decode_benchmark/runs/gst_rpt4/summary.md): rpt4 에서 DMABuf 381.5 FPS / 0.24 ms/f. 회차 [`runs/ffmpeg_outfmt`](decode_benchmark/runs/ffmpeg_outfmt/summary.md): CLI 2.0 ms 중 약 1.5 ms 가 SAND→planar 내려받기, DRM_PRIME 유지 시 386.9 FPS / 0.55 ms/f, `vf_unsand` 는 이 빌드에서 실행 불가. (앱에 직접 넣은 libavcodec DRM_PRIME 은 **398.5 FPS / CPU 0.23~0.28 ms/f**(프레임 스레드 자동 5개 조건. 앱 기본값인 1스레드는 약 266 FPS 로 코드 주석 기준, 미측정) — 이 값의 출처는 이 폴더가 아니라 [`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.1 이다) |
| [`vulkan_compute_codecs/`](vulkan_compute_codecs/summary.md) | 프로브 | 유효 | 2 디코드 | — | FFmpeg 8.1.3 Vulkan compute 코덱(FFv1·ProRes·DPX 디코드, ffv1/prores_ks 인코드)을 V3DV·llvmpipe 에서 실행(프로젝트 외 조사) | **V3DV 불가.** FFmpeg semaphore export 버그(시스템 7.1.5 포함) + 셰이더의 Int64·포맷 없는 image store·runtime descriptor array 를 V3D 가 미지원 → 컴파일러 abort·행·조용한 오출력. llvmpipe 는 FFv1 range·DPX 바이트 동일, ProRes ±1 LSB 이지만 CPU 보다 12~170배 느림 |
| [`isp_resize/`](isp_resize/summary.md) | 스윕 | 보류 | 3 전처리(ISP) | `runs/{pipeline,pipeline_rerun}` | PiSP BE(ISP) 전처리 vs GPU 셰이더 | ISP 가 wall 로 23% 빠르고 V3D 를 비운다. **단 검출 −2.33% 로 게이트 불합격 → 통합 보류.** 재검토 조건 3개 명시(그중 하나가 CSI 카메라) |
| [`pipeline_ffmpeg_vulkan_hailo/`](pipeline_ffmpeg_vulkan_hailo/summary.md) | 스윕 | 유효 | 4 통합 | — | 디코드 → GPU 전처리 → NPU 통합 | **51.63 FPS**(NPU 상한의 99.9%). **NHCW 직접 출력으로 CPU 17.6 → 3.5 ms/f**(디코더 자동 스레드 조건. 1스레드 기본값에서는 nhcw b8 2.93 ms/f, [`dec_threads/`](dec_threads/summary.md)). 비싼 것은 전치가 아니라 **CPU 가 V3D 메모리를 읽는 것**이었다 (상세: [`REPORT.md`](pipeline_ffmpeg_vulkan_hailo/REPORT.md)). 2026-09-29 회차 [`runs/bootloader_20251208`](pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20251208/summary.md) → [`runs/bootloader_20260925`](pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md): 부트로더 갱신(V3D 960 → 1150 MHz)에도 FPS·CPU 불변, **GPU 전처리 b8 2.6 → 2.0 ms, p90 5.5 → 2.1 ms**(느린 꼬리 소멸) |
| [`gst_pipeline/`](gst_pipeline/summary.md) | 스윕 | 유효 | 4 통합 | — | 앱(libavcodec DRM_PRIME) 대 GStreamer 연구 경로(`v4l2slh265dec` DMABuf → `yvosandletterbox` → `yvohailosink`), b1/b4/b8 같은 slots | **검출 `detections.jsonl` 바이트 동일**(9457), FPS 동일(NPU 상한), **CPU 는 gst 가 약 0.2 ms/f(6.3%) 낮음**(당시 앱은 디코더 자동 스레드 — 원인은 아래 `perf_dec_threads` 회차가 확정). rpt4 이후 성능상 libavcodec 을 고집할 이유는 없어짐 — 대신 패키지 버전 의존. 측정 구간·e2e 시작점 정의가 다르다. 회차 [`runs/perf_breakdown`](gst_pipeline/runs/perf_breakdown/summary.md): 같은 구간 perf — 명령어 수 동일, 차이의 ~70% 는 **HailoRT 호스트 NMS 점수 스캔 루프가 앱에서 더 느린 것**(L3 미스 수는 같고 미스당 지연이 김), 나머지는 FFmpeg HEVC 프레임 스레드의 memset·문맥 교환. 디코드 코드 자체는 원인 아님. 회차 [`runs/perf_dec_threads`](gst_pipeline/runs/perf_dec_threads/summary.md): **앱의 FFmpeg 프레임 스레드를 1개로 두면 CPU 차이 74% 감소·cycles 는 gst 이하**(FPS·검출 동일) → 원인은 FFmpeg 프레임 스레딩 확정. 단 NMS 루프 차이는 재현 안 됨(회차 간 변동) — perf_breakdown 의 NMS 메커니즘은 대체됨. 이후 앱 기본값이 1스레드가 돼(5032575, [`dec_threads/`](dec_threads/summary.md)) 차이의 대부분이 해소됐다 |
| [`dec_threads/`](dec_threads/summary.md) | 스윕 | 유효 | 4 통합 | — | 앱 FFmpeg HEVC 프레임 스레드 수 스윕(1~4·자동)과 자동 대 1스레드 회귀 A/B(ptr·nhwc·nhcw b1/b4/b8, GPU 후처리, 지연 운용점, 표시 on) | **1스레드가 CPU −14%·e2e −8.8 ms(b8)**, 8개 모드 FPS·검출 동일, CPU·e2e 평균 모두 개선. 예외: 표시 on(source pace) 앱 내부 e2e p90 +1.9 ms(glass 평균 동일) |
| [`nms_host_cost/`](nms_host_cost/summary.md) | 스윕 | 유효 | 5 후처리(호스트) | — | 호스트 NMS 비용 | `yolov8_nms_postprocess` 는 **온칩이 아니라 호스트 CPU** 실행. **2.2~3.2 ms/f 고정비**(검출 수와 무관). 원시 출력으로 바꿔도 **PCIe 트래픽은 안 늘어난다** |
| [`gpu_postprocess/`](gpu_postprocess/summary.md) | 프로브 | 유효 | 6 후처리(GPU) | [`local_sweep_argmax`](gpu_postprocess/runs/local_sweep_argmax/summary.md) | GPU 후처리 프로브 (P1/P2/P3) + **[F0] 관문** | 텐서 해석 4변수 전수 확정. pass1 **0.5625 ms**(V3D 960 MHz), CPU 억제 3~11 µs. **[F0]: 입력·출력 dma-buf 동시 사용 동작 확인**(`evidence/f0_in_dmabuf.txt`). 회차 `local_sweep_argmax`(2026-10-11, 1150 MHz): argmax 지름길 뒤 재스윕 — **10×6 유지**. 3% 기준을 넘는 조합 없음(가장 가까운 mot20_05의 5×12도 −1.8~−2.1%이고 다른 입력에선 +1.9~8.2% 느림). 순위 구조가 2026-09와 같아 최적점이 옮겨지지 않았다. 960 MHz 표와 절대 ms 비교 불가 |
| [`display_vulkan_hdmi/`](display_vulkan_hdmi/summary.md) | 프로브 | 유효 | 7 표시 | — | 표시 프로브 (P4~P10) | 컴퓨트가 **스왑체인에 직접 STORAGE 쓰기** 가능(blit 불필요). 1080p 1:1 **4.229 ms**(V3D 960 MHz). ⚠ 창 모드는 **조용히 2.6배 느려진다**. 후속 메모(각 1회·비정식): [`notes/kms_vs_wayland.md`](display_vulkan_hdmi/notes/kms_vs_wayland.md)(HVS 두 평면 스캔아웃 KMS 대 Wayland, 증거 `evidence/hvs_review/`, §4.1 = 2026-10-07 색 범위 재판정 · 원시 `verification/color_range_rpt4/`), [`notes/wayland_planes_pacing_gate.md`](display_vulkan_hdmi/notes/wayland_planes_pacing_gate.md)(커밋 간격 게이트). summary §P10·결론의 "안 (III) 미검증"은 이후 `wayland-planes`/`kms` 로 구현됐다 |
| [`dmaheap_unify/`](dmaheap_unify/summary.md) | 프로브 | 별도 트랙 | — 별도 트랙 | — | dma-heap → V3DV import 관문 (별도 트랙) | V3DV 가 dma-heap fd 를 import 한다. **단 `range = size − 256`** 함정. GPU 읽기 성능(B.2)은 **닫혔다**(heap/plain = 0.9994, 손해 없음). Hailo 방향 3자 체크섬(V1, 120건)은 불일치 0 PASS. **열린 것**: V3D 방향 캐시 일관성 기제 미확정, sync 필요성 미실증(sync 유지), `--alloc heap` 경로가 ctest 에 없음 |
| [`op_point/`](op_point/summary.md) ⭐ | 스윕 | ⭐최종 | 통합 2026-09-20 | `runs/{latency_gen3_cooled,latency_sweep,grid_gpu_postproc,latency_b1s1_20261012}` | **운용점 결정 (O1~O4)** + 냉각 Gen3 지연 재측정 + 지연 스윕 + 현재 기본값 batch×slots 격자(2026-10-08) | **이 폴더가 최종 결론이다(§8).** slots 가 지배 변수임을 정량 확인. `b4/s8` vs `b8/s8` 동률로 계획서 F1 가설 기각. 격자 재측정([`grid_gpu_postproc`](op_point/runs/grid_gpu_postproc/summary.md)): 운용점 불변, FPS = min(배치 천장 `B/T(B)`, slots 제약)이고 천장은 slots = 2B 에서 닿는다. 회차 [`latency_b1s1_20261012`](op_point/runs/latency_b1s1_20261012/summary.md)(현재 HEAD, 최저 지연 b1/s1): **e2e 79.19 ms · p90 81.8 · 24.42 FPS 로 10-08 과 같다**, host 대 GPU 후처리 e2e 같음(CPU 3.25 → 0.94 ms/f), gpu-depth 1 = 3. 절반 넘게(약 42 ms)가 슬롯 대기. **O2(최소 지연)는 2026-10-12 [`input_pacing/`](input_pacing/summary.md) 의 페이싱 b1/s1 로 바뀌었다** — 이 폴더 §5.5·§8 의 O2(81.6 ms)는 페이싱 없는 기록이다 |
| [`gpu_postproc_e2e/`](gpu_postproc_e2e/summary.md) | 스윕 | 일부 인용금지 | 통합 2026-09-20 | — | `--postproc gpu` 통합 정확도·CPU | **매칭 @ IoU 0.9 = 0.9990, 검출 9457 로 host 와 완전 동일.** perf 에서 호스트 NMS 45.10% → **0건**. ⚠ 이 폴더의 FPS·지연은 **결함 상태 값이라 인용 금지**(문서 안에 표시) |
| [`raw_runner_backpressure/`](raw_runner_backpressure/summary.md) | 스윕 | 유효 | 통합 2026-09-20 | — | 러너 백프레셔 이중 계수 결함 규명·수정 | 수동 `increase_ongoing_callbacks()` 가 `infer_async` 내부 계수와 겹쳐 **in-flight 가 슬롯의 절반**. 제거 후 gpu `b2/s4` 24.48 → **34.96 FPS**(host 대비 −0.03%) |
| [`gpu_nms_e2e/`](gpu_nms_e2e/summary.md) | 스윕 | 유효 | 4 통합 | [`argmax_shortcut_mot20_05`](gpu_nms_e2e/runs/argmax_shortcut_mot20_05/summary.md) · [`head_20261012`](gpu_nms_e2e/runs/head_20261012/summary.md) | **GPU NMS 정식 A/B + 현재 기본값 기준선**(표시 off, V3D 1150 MHz·디코더 1스레드, b2s4·b8s16 × `--nms cpu|gpu`, 900f×3) + 전체 영상 정확도 + NMS 단독 GPU 비용 | **회귀 없음**: FPS·검출 동일, CPU·e2e 차이는 실행 간 범위 안(CPU NMS 원가가 약 2 µs/f라 절감은 안 보임). 정확도 **바이트 동일**(9457, verify 불일치 0, fallback 0). GPU NMS 0.049~0.050 ms/f(프레임당 CSD 잡 +1), 후보 1024 최악 3.05 ms. **현재 기본값 기준선: b8s16 51.50 fps · CPU 0.869 ms/f**, b2s4 34.99 fps(운용점 b2/s4 의 파이프라인 처리량 — slots 가 정한다, `hailortcli` 의 T(B) 아님) · 0.908 ms/f. 회차 `argmax_shortcut_mot20_05`(2026-10-11): argmax 지름길 종단간 — 실영상 900프레임 검출이 지름길 이전과 **바이트 동일**(교차 커밋 대조 3쌍, `[A2] PASS`만으로는 증거가 안 된다 — 양쪽이 같은 후보를 받는다). **V3D CSD −10.3~−11.4%(약 0.29 ms/f)**, GPU 후처리 −32~−35%, FPS·지연 불변(**FPS로 주장하지 않는다**). 회차 `head_20261012`(2026-10-12, HEAD `85a0d9a`): 2026-10-11 리팩터 4건 뒤 표시 off 재측정 — **회귀 없음**(b8s16 51.51 fps · CPU 0.909 ms/f, b2s4 34.99 · 0.915, 둘 다 CPU 3회 범위가 기준선과 겹침), 전체 영상 검출 기준선과 **바이트 동일**, 기본 영상에서도 GPU 후처리 −15% |
| [`nms_crowd_stress/`](nms_crowd_stress/summary.md) | 프로브 | 유효 | 4 통합 | `runs/{mot20_02,mot20_05,mot20_05_lowthr}` | **혼잡 장면 스트레스 시험**(MOT20 영상 2개, 1080p25 900f): 원시 텐서 픽스처 대조 + `nms_bench.sh` perf/verify(b2s4·b8s16 × `--nms cpu|gpu`, 900f×3) + 점수 임계값 스윕(b8s16, 0.2~0.003) | 후보가 기존의 약 11배(평균 180~183·최대 267)여도 **FPS 그대로, 검출 CPU·GPU 바이트 동일, CPU로 넘긴 프레임 0**. GPU NMS 0.05 → 0.18 ms, pass1 0.55 → 0.77~0.89 ms. 1024 초과(점수 ≤ 0.02에서만)는 설계대로 CPU 처리, 프레임당 약 0.8~1.5 ms(최대 2.8), 처리량 영향 없음. **4096 넘침 시 앞 4096개만 읽어 검출을 조용히 잃고 verify도 못 잡는다**(버퍼를 1024로 줄인 재현에서 829개 중 6개 손실) — 지금 조건으로는 양자화 바닥(점수 0.003, q ≥ 1)에서도 최대 1925개라 닿지 않는 잠재 결함. 클래스당 100 한도·IoU 보호대는 한 번도 작동 안 함. 혼잡의 실제 한계는 재현율(주석 73~227명 대 검출 21~26명) |
| [`postproc_stage_cost/`](postproc_stage_cost/summary.md) | 프로브 | 유효 | 6 후처리(GPU) | [`emit_split`](postproc_stage_cost/runs/emit_split/summary.md) · [`emit_shortcut`](postproc_stage_cost/runs/emit_shortcut/summary.md) · [`rescan_causes`](postproc_stage_cost/runs/rescan_causes/summary.md) | **GPU 디코딩 단계별 비용**(`yolo_decode.comp` ①~⑤, 단계 멈춤 변형 차분법, `--probe stages`) + 같은 픽스처 NMS. 기존 영상·MOT20 픽스처, 점수 0.2~0.003 | **④ 레터박스 역변환은 0.001~0.007 ms — NMS 뒤로 옮겨도 이득 없음**("NMS 먼저" 제안의 상한 ≈ 0). ①② 클래스 스캔은 후보와 무관한 고정 약 0.30 ms(기존 영상 pass1의 60%). 후보 수에 비례하는 것은 ⑤ emit(혼잡 점수 0.2에서 pass1의 41~45%, 후보당 0.5~1.6 µs)과 ③ DFL. 혼잡 장면 디코딩+NMS 약 1 ms/frame. 회차 `emit_split`(2026-10-11): **⑤의 94.4~97.8%가 클래스 재스캔이고 atomic 몫은 0**(−0.008~+0.002 ms) — 모 회차가 남긴 "atomic을 워크그룹 단위로 모으기" 제안은 지렛대가 아니다. 그룹당 통과 앵커 평균 1.37~1.74라 클래스 외곽 루프 교환의 상한은 ⑤의 26~41%뿐이고, **앵커당 후보가 "정확히 1개"가 100.00%**. 회차 `emit_shortcut`(2026-10-11): ①에서 argmax를 패킹해 들고 나오는 지름길로 **운용 임계값 0.2에서 pass1 −14.0~−36.3%**(⑤ 0.126→0.040 · 0.310→0.051 · 0.367→0.052), 범위 비중첩. 대가는 ①의 고정비 **+0.039 ms**(일정). **검출 불변 — 15회 전부 다중집합 불일치 0건.** thr 0.003은 `q_thr_low=0`이라 전부 폴백이어서 +0.05 ms 회귀(운용점 아님). ⚠ 이 행 앞부분(①② 0.30 ms 고정, ⑤가 후보 비례)은 **지름길 이전** 기록이다 — 지금은 ①② 약 0.34 ms, ⑤ 0.04~0.05 ms(운용 임계값 0.2 기준). 조건은 V3D 1150 MHz이고 **FPS로 주장하지 않는다**(의미는 V3D CSD 예산의 여유). 회차 `rescan_causes`(2026-10-11, 오프라인 재현·GPU/NPU 미사용): **지름길이 못 받는 앵커(폴백)는 임계값 0.2에서 0건**(⑤ 진입 3871개 전부 지름길). 0.05·0.02에서 0~16.4%이고 대부분 비슷한 클래스(car↔truck, handbag↔backpack. 예외 mot20_05 0.02는 42%), 양자화 헛경보는 폴백의 많아야 약 28% |
| [`candidate_reduction_survey/`](candidate_reduction_survey/summary.md) | 프로브 | 유효 | 1 NPU | — | **후보 줄이기(점수·topK)로 PCIe·NMS 부하를 줄일 수 있나** — 현재 HEF의 PCIe 바이트 회계(HEF 디코드·estimator·fw-actions), 칩 안 후보 축소 가능성(Model Zoo·HailoRT 소스·커뮤니티), 호스트 topK 정리 | **경계 출력 1.22 MB는 프레임당 전송의 약 6%이고 빈 D2H 방향을 타서, 0으로 만들어도 처리량 +2.3% 이하**(천장 6.2%). 칩 NMS는 yolov8 메타라 불가, topK·가변 길이 출력도 불가. T(B) 의 고정비 24.8 ms ↔ config 20.5 MB 재적재, 프레임당 16.2 ms ↔ 컨텍스트별 max(H2D,D2H) 12.84 MB(실효 0.80 GB/s)로 **바이트와 정합**. 진짜 지렛대는 config 재적재와 컨텍스트 사이 텐서의 호스트 왕복(13.8 MB/프레임) |
| [`hef_context_profile/`](hef_context_profile/summary.md) | 프로브 | 유효 | 1 NPU | — | **HEF 컨텍스트별 NPU 시간 분해**(2026-10-12) — `hailortcli run2 -m raw_async measure-fw-actions`, B = 1·2·8 × 3, 펌웨어 타임스탬프(200 MHz)로 컨텍스트 4개의 구간·액션 종류별 몫·지연(마지막 출력까지)과 주기 분리, 9회 `a + b·B` 적합 | **배치 구간 `24.79 + 16.23·B`(rms 0.27)로 T(B) 재현.** 고정비 최대는 **dynamic_3(10.03 ms, 40%)**, 다음 dynamic_1(7.40, 30%). 각 컨텍스트가 다음 컨텍스트의 config를 미리 적재하는 것으로 보이며(추정) dyn3은 마지막 출력 뒤 다음 배치 dyn0 config(8.15 MB)를 기다린다(b1 꼬리 8.35 ms). 컨텍스트마다 고정 a ↔ config 바이트(합 0.83 GB/s), 프레임당 b ↔ max(H2D,D2H)(0.79 GB/s). **O2(b1) 칩 지연 32.57 ms = 고정 15.71(48%) + 프레임당 16.86** — 파이프라인 NPU 34.6 ms의 45%. 주기 41.1 ms 중 고정 24.8(60%)이 30 fps 미달·버림 18.6%의 원인. 컨텍스트 전환 자체는 0.01 ms라 이득은 바이트에서만 나온다(config 1.21 ms/MB, 왕복 1.26 ms/MB). 단일 컨텍스트 극단 상한 칩 지연 −15.7 ms·b1 ≤ 61 FPS(모델 교체 = 사용자 결정). JSON `fps`는 중간 표본이라 인용 금지(stdout 사용) |
| [`topk_offline/`](topk_offline/summary.md) | 프로브 | 유효 | 6 후처리(GPU) | — | **후보 축소 방식 오프라인 비교** — 앵커별 max(`max(-1)`)·topK(임계값 없음/있음)·다중 클래스 유지 컷(MC_HIST)을 현재 기준(T 이상 모든 클래스)과 비교. `--probe topk`(CPU 전용, 기준 디코드 재사용), 영상 3개 × 앞 60프레임, T 0.2~0.003, K 100/300/1024 | **임계값 없는 topK는 결과를 망친다**(한산한 장면 검출 101 → 4495, 혼잡 K=100에서 26~38% 손실). 임계값을 더해도 운용값 0.2에서는 K ≥ 300이어야 손실 0이라 줄일 것이 없다. `max(-1)`은 0.2에서 1% 이하, 낮은 T에서 9~25% 검출 변화. 1024 용량 안전장치로 쓰면 T=0.02에서 8.7% 손실 — **채택하지 않음** |
| [`display_planes_e2e/`](display_planes_e2e/summary.md) | 스윕 | 유효 | 7 표시 | `runs/slots6` | **HVS 평면 표시 정식 측정**(실험 B): Vulkan 합성 / `wayland-planes` / `kms` × pace source·asap, b2/s4, 추론 on, 900f×3, 앞 30 프레젠테이션 제외 | 평면 경로가 **CPU −0.49 ms/f(2.059 → 1.569 / 1.576, −24%)**, **V3D 10.19 → 2.89 ms/f(−72%, TFU 3.72 ms 소멸)**. wayland-planes 와 kms 는 CPU·V3D·glass·늦은 프레임이 실행 간 범위 안에서 겹친다(kms 의 labwc CPU 0 은 데스크톱 비활성 때문). 색·zero-copy·underrun·flip 검사 12/12 통과, 검출 3369 동일. **⚠ b2/s4 에서는 평면 경로가 버퍼를 release 까지 쥐어 슬롯이 모자라 디코드 후 평균 28~30 ms 를 기다린다** — glass(슬롯 획득 뒤부터 잼) 99.9 ms 에 안 들어가며, 디코드 → 화면은 vk 약 102 대 평면 약 130 ms. 회차 [`runs/slots6`](display_planes_e2e/runs/slots6/summary.md)(b2/s6): 슬롯 대기 0, **디코드 → 화면 평균 vk 101.4 / wlp 102.3 / kms 101.6 ms 로 같다 — 평면 경로의 이득은 지연이 아니라 CPU(−0.44 ms/f)·V3D(−72%)뿐이고, 표시 on 평면 경로는 b2/s6 이 필요하다.** source 페이싱에서 프레임의 약 25~27%가 한 vblank 늦다(슬롯과 무관) |
| [`display_e2e/`](display_e2e/summary.md) | 스윕 | 일부 인용금지 | 통합 2026-09-20 | — | 표시 통합 e2e (M5a 기준선 + M4) | **glass-to-glass 101 ms**(계획서 추정 217 의 절반). GPU 4패스 합 **6.995 ms = 주기의 24.5%**(V3D 960 MHz·자동 스레드·CPU NMS). 표시 켜도 **검출 3369 불변**, 드롭 0 |
| [`v3d_perf_counters/`](v3d_perf_counters/summary.md) | 프로브 | 유효 | — 프로파일링 도구 | — | V3D `VK_KHR_performance_query` 카운터 동작 특성 (E1 분해 단위 · E2 혼입 · E3 오버헤드) | 카운터 93개, 1 pass 32개. **한 쿼리는 안의 모든 잡을 합산**(k배 정확). **다른 클라이언트의 bin/render 잡이 겹치면 값이 잘리고 섞인다**(vkcube 중 정상의 1.4~4%, 유휴 데스크톱도 100회 중 0~5회) → 표시 off·중앙값·최빈값 이탈 표본 제외. GPU 쪽 쿼리 비용은 잡음 안(+7 µs/4.9 ms). 회차 [`runs/bootloader_20260925/`](v3d_perf_counters/runs/bootloader_20260925/summary.md)(V3D 1150 MHz): 결론 유지, heap_sum 은 클럭 1.198배에 1.10~1.13배만 빨라짐(메모리 대기), **실행 중 +2.6~6.3% 계단**(원인 미확인) |
| [`thermal_soak/`](thermal_soak/summary.md) | 프로브 | 유효 | 통합 | — | **연속 운용 발열**(2026-10-11) — `b2s4`·`b8s16` 각 40분, 혼잡 영상 반복 입력, 차갑게 시작, 추가 냉각 없음. 구간 FPS(Δ프레임/Δ시각)와 HailoRT 클럭·온도 zone 경고 시각을 대조 | **`b2s4`는 지속 가능** — 325초에 orange zone(103.98 °C), 이후 칩 101~108.5 °C·클럭 300↔350 MHz에서 평형, 구간 FPS 35.05 → 평형 **34.25~34.75**(최저 −2.3%), 30 fps 대비 최저 +14.2%, 40분 완주·에러 0. **`b8s16`은 지속 불가** — 110초에 orange, 클럭 200 MHz까지 내려가도 평형 없이 상승, **316초에 119.97 °C red zone → 헬스 모니터가 스트림을 닫아 추론 중단**, 정지 전 51.74 → 44.86 FPS(−13%). 클럭 25% 하락에 FPS 2.3%만 → 연산은 배치 주기의 약 7%, 나머지는 PCIe(열 조건에서도 PCIe 지배 확인). SoC 최고 75.2 °C(`b8s16` 유효 구간 74.1 °C)·스로틀 0 — 열 한계는 Hailo 칩 냉각 |
| [`input_pacing/`](input_pacing/summary.md) ⭐ | 스윕 | ⭐O2 근거 | 통합 | [`frame_log`](input_pacing/runs/frame_log/summary.md) · [`display_on`](input_pacing/runs/display_on/summary.md) | **입력 페이싱 + leaky 입력 큐**(2026-10-12) — `--input-pace source`(프레임이 30 fps 시각에 도착) + `--input-queue N`(GStreamer `leaky=downstream`, FIFO·가장 오래된 것 버림) N=1·2·3 × b1/s1·b2/s3·b2/s4, asap 대조군, 도착 900프레임 × 3, 표시 off. 지연은 도착 시각부터(`e2e_arr`) | **최소 지연 운용점 O2 의 근거**(사용자 결정, 2026-10-12): b1/s1 + 큐 1 은 도착 기준 평균 58.3 · p90 72.7 ms, 버림 18.6%. **페이싱만으로 30 fps 추종 설정의 지연이 평균 81.3(b2/s4)·81.6 ms(b2/s3), p90 104 ms**(asap 디코드 기준 128~140 ms), 버림 0. **b2/s3 = b2/s4**(큐가 비어 차이 소멸). **N 은 b2 에서 효과 없음**(큐 최대 깊이 늘 1 — 빈 슬롯이 지터를 흡수, N=1 의 지터 약점은 나타나지 않음). b1/s1 은 N 한 단계마다 **+33 ms**(58.3 → 91.4 → 124.6), 버림 18.6% 고정 → **기본 N=1**. 회차 `frame_log`(2026-10-12, `--frame-log`): 페이싱 b2 는 NPU 실행의 55% 가 **1장 단독**(asap 은 거의 전부 2장), 짝의 첫째만 약 100 ms. b2 max 136~152 ms 는 **실행 끝 drain 현상**(정상 상태 최대 약 107 ms). O2 58 ms = NPU 37.6 + 큐 대기 14.5 + 후처리 관찰 3.4 + 도착 처리 2.6. 회차 `display_on`(2026-10-12, 표시 on, 15회 유효): **도착 → 화면** O2 Vulkan 평균 73.1 · p90 82.6 ms(버림 22.2% — slots 1 이 표시 fence 를 기다려 주기 40.9 → 42.3 ms), 평면 b1/s2 68.4 ms 지만 처리가 vblank 3개(50 ms)에 잠겨 20 fps · 버림 33%, 평면 b1/s3 90.8 ms · 24.4 fps, b2/s4 Vulkan 103.3 · b2/s6 평면 104.1 ms(버림 0, 평면이 꼬리 짧고 간격 고름) |

## 폴더 규약

레이아웃·evidence/verification 경계·이동 원칙의 정본은 [`STRUCTURE.md`](../STRUCTURE.md) §3 이다. 요점만:

- 공통 필수는 `summary.md`·`env_bench.txt`. 실행 로그는 `raw/<설정>_run<N>/`, 세션 부산물은 `raw/_session/`.
- 같은 실험의 회차·변종은 `runs/<회차>/` — 새 실험이 아니다(위 표의 "회차" 열).
- 결과 파일의 **내용**은 고치지 않는다. 위치를 옮길 때는 `relocate.py` 와 `STRUCTURE.md` §6 매핑표를 쓴다(2026-09-22 R1 이동 이력이 거기 있다).
- 구조 검사: `python3 06_tools/python/check_structure.py`

### 중복 산출물

2026-09-21 에 바이트 일치 PNG 22장(7.4 MB)을 정본만 남기고 지웠다(46장 16.2 MB → 24장 8.8 MB).

**지우기 전에 md5 대조표를 남겼다**: [`duplicate_png.md`](duplicate_png.md).
몇몇 실험은 *두 경로의 산출물이 바이트까지 같다*는 것 **자체가 결론**이기 때문이다 —
host 경로 == GPU 경로, ISP A안 == C안, batch 4 == batch 8, NHWC == NHCW.
표의 md5 로 지워진 파일이 남은 파일과 같은 내용이었음을 확인할 수 있고,
바이트 일치 `detections.jsonl` 4개(3.0 MB)도 같은 표에 기록하고 지웠다. **모든 jsonl 은 정본 1부가 남아 있다**(PNG 는 그것의 시각화였다).

> 앞으로도 같은 순서로 한다: **md5 로 동일성을 확인 → 대조표에 기록 → 삭제.**
> 인용된 파일은 지우지 않는다.

문서 링크가 깨지지 않았는지는 이걸로 확인한다:

```bash
python3 06_tools/python/check_links.py
```

---

## 읽는 순서

- **"지금 뭘 쓰면 되나"** → [`op_point/summary.md`](op_point/summary.md) §8
- **"왜 이 설계인가"** → `pipeline_ffmpeg_vulkan_hailo/REPORT.md` §4.3 (메모리 제약) → `nms_host_cost/` (후처리 비용)
- **"표시가 어떻게 동작하나"** → `display_vulkan_hdmi/` (프로브) → `display_e2e/` (통합) → `display_vulkan_hdmi/notes/` (HVS 평면 스캔아웃, 비정식) → `display_planes_e2e/` (평면 표시 정식 측정)
- **"정확도를 어떻게 보장하나"** → `ctest` 177개. 특히 `test_postproc`(`PostprocFixture` 6 — heap 입력 경로 포함 + `EfficientNms` 7 + live 1), `test_display_convert`,
  `test_detections`, `test_preproc`, `test_overlay`. 판정 규칙은 `01_src/ref/detection_match.hpp` 하나로 공용화돼 있고
  `06_tools/python/compare_detections.py` 가 같은 규칙을 구현한다

## ⛔ 인용 금지 구간

**이 절이 정본이다.** 아래 두 구간은 이 프로젝트의 어떤 문서·논문·보고에서도 인용하지 않는다.

1. **백프레셔 결함 수정 이전의 GPU 후처리 경로 처리량·지연.**
   근거: [`raw_runner_backpressure/summary.md`](raw_runner_backpressure/summary.md).
   [`gpu_postproc_e2e/`](gpu_postproc_e2e/summary.md) 의 FPS·지연 값이 여기 해당한다
   (같은 폴더의 **정확도·perf 결과는 유효하다**).
   결함을 *서술하는* 맥락에서 "수정 전 24.48 → 수정 후 34.96" 처럼 쓰는 것은 인용이 아니라 설명이므로 허용한다.
2. **`--display-pace asap` + 표시 on 조건의 GPU timestamp `mean`.**
   V3D 가 연속 제출된 커맨드 버퍼를 병합해 한쪽 구간이 0에 가깝게 기록된다(**원인 미확정**).
   이 조건에서는 **중앙값만** 쓴다. 근거: [`display_e2e/summary.md`](display_e2e/summary.md).

> 이 규칙의 정본은 원래 `10_paper/paper_ko.md` 부록 A.4 였다. 논문은 저장소 밖으로 나가거나
> 개정될 문서이므로 2026-09-21 에 이리로 옮겼다. 논문 A.4 는 이제 이 절을 참조한다.

---

## 아직 열려 있는 것

**이 표가 정본이다.** 루트 `README.md` 와 `08_learn/10_why_this_design.md` 는 여기를 링크한다.
보류·비수정으로 **닫은** 항목은 결정의 **주체와 날짜**를 함께 적는다 — `(사용자 결정, YYYY-MM-DD)`.
주체 없는 기록은 나중에 근거로 쓸 수 없다(`CLAUDE.md` §사용자 확인). 이미 적힌 행은 그대로 둔다.

| 항목 | 왜 아직인가 |
|---|---|
| **연속 운용 발열** ★ (일부 닫힘) | [`thermal_soak/`](thermal_soak/summary.md)(2026-10-11)가 답했다 — **`b2/s4`는 40분 지속 가능**(평형 34.25~34.75 FPS, 최저 여유 +14.2%), **`b8/s16`은 추가 냉각 없이 지속 불가**(5분 16초에 red zone 정지). 예전 가정("90 °C대에서 400→250 MHz")은 틀렸고 실제로는 약 104 °C에서 단계적으로 내려가 약 120 °C에서 정지한다. **남은 것**: 중간 운용점(`b4/s8` 등)의 지속 여부, 추가 냉각을 단 상태, 표시 on, 40분 초과, 주변 온도 조건 |
| **카메라(CSI) 입력 경로** | 디코더가 `01_src/decode/ffmpeg_drm_decoder.cpp:166` 에서 **HEVC 파일만** 받는다. 카메라로 가면 ISP 경로 재검토 조건도 함께 열린다 |
| `--display-sync latest` 정식 측정 | 150프레임 확인 실행만 했다(glass median 28.2 ms) |
| dma-heap 전환 (B.2 성능·V1 체크섬은 닫힘) | 별도 트랙. 성능(heap/plain 0.9994)과 Hailo 방향 3자 체크섬(V1 PASS)은 [`dmaheap_unify/`](dmaheap_unify/summary.md) 에서 확인됐다. ctest 공백은 **닫혔다(2026-10-11)** — `PostprocFixture.ImportHeapMatchesPlainBitExact` 가 heap 입력 경로를 Plain 과 비트 일치로 고정하고, 헤드룸 없는 버퍼가 실제로 잘리는지(range = size − 256 함정)를 매 실행 음성 대조로 확인한다. 남은 것: 현재 이득 0 이라 기본 경로는 전환하지 않았다. `--postproc-alloc heap` 은 옵션만 있고 **기본값은 `export`** 다. V3D 쪽 기제는 아래 별도 행 |
| 다중 스트림 | **측정 불필요** — 1080p30 2개 = 60 FPS 인데 NPU 상한이 51.7 이다 |
| 워크그룹 최적값의 **이유** | `yolo_decode` 가 60 invocation(10×6)에서 256보다 빠르고 `display_convert` 가 3×85 에서 16×16보다 1.83배 빠른 이유를 설명하지 못했다. **측정치만이 근거인 가설**이다 |
| V3D 쪽 캐시 일관성 **기제** | Hailo 쪽은 드라이버가 관리함을 perf 로 확인했다. V3D 는 프레임마다 매핑하지 않는데도 오염이 관측되지 않는다 — 축출 완료인지 스누핑인지 **가르지 못했다**(결정적 실험에 root 필요) |
| `pace=asap` 의 GPU timestamp 평균 왜곡 | 커맨드 버퍼 병합으로 **추정하나 미확인**. 해당 값은 위 "인용 금지 구간" 2번이다 |
| GPU NMS·HVS 평면 표시 정식 측정 | **닫혔다(2026-10-08).** GPU NMS 는 [`gpu_nms_e2e/`](gpu_nms_e2e/summary.md), 평면 표시는 [`display_planes_e2e/`](display_planes_e2e/summary.md) + 회차 `runs/slots6`. 남은 것은 표시 박자 하나다: source 페이싱에서 프레임의 약 25~27%가 한 vblank 늦게 나간다(wlp·kms 같음, 슬롯과 무관, 원인 미검증 — vblank 위상 추정). Vulkan 합성도 present p90 66.8 ms 로 꼬리가 있다 |
| 디코딩 후보 4096 넘침 시 조용한 잘림 | `read_candidates()` 가 앞 `max_dets`(4096)개만 읽어 넘친 프레임의 검출을 잃고, verify 도 같은 잘린 후보로 비교해 못 잡는다. 지금 HEF·영상으로는 양자화 바닥에서도 닿지 않지만(최대 1925) 다른 모델이나 더 혼잡한 장면에서 드러난다. **현 HEF·입력에서는 발생할 수 없어 수정하지 않고 남긴다(2026-10-08 결정).** 모델을 바꾸면 다시 본다 — 후보 수정안(원시 텐서 CPU 전체 디코딩 / `max_dets` 상향 / 넘친 프레임 표시) — [`nms_crowd_stress/`](nms_crowd_stress/summary.md) §4 |
| KMS 종료 시 이전 상태 복원 | 복원 커밋이 `EINVAL` 로 실패하고 평면 끄기로 폴백한다(labwc FB 가 바뀐 것으로 추정, 미확인). 데스크톱 복귀에는 영향 없음 |
| Vulkan present 무기한 대기 | 컴포지터가 frame 콜백을 주지 않으면(labwc headless) 스왑체인 경로가 `vkQueuePresentKHR` 안에서 멈추고 SIGTERM 도 안 먹는다. 원인(Mesa WSI 대기)은 추정. 평면 백엔드는 해당 없음 |
| labwc COLOR_RANGE 뒤집힘 의존 | `wayland-planes` 는 full 을 요청해 limited 를 얻는 우회("fullhack")에 기댄다. 컴포지터가 고쳐지면 조용히 틀어지므로 debugfs 판정(`plane_color_ok`)을 매번 본다. 2026-10-07 labwc 0.20.2 / wlroots rpt4 업그레이드 뒤 재판정 — 반전 그대로, 코드 변경 없음([`display_vulkan_hdmi/verification/color_range_rpt4/`](display_vulkan_hdmi/verification/color_range_rpt4/README.md)) |
| 커밋 간격 게이트 여유 8 ms 의 최적값 | `wayland-planes`/`kms` 의 pace=source 게이트(소스 간격 − 8 ms)를 튜닝하지 않았다 |
| 960 MHz 시절 b4/b8 GPU 느린 꼬리의 원인 | 부트로더 2026-09-25(V3D 1150 MHz) 뒤 p90 5.5 → 2.15 ms 로 사라졌지만 무엇이 꼬리를 만들었는지는 가르지 않았다([`bootloader_20260925`](pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md)) |
| 성능 카운터의 +2.6~6.3% 계단 | 1150 MHz 회차에서 실행 도중 GPU 작업 시간이 계단처럼 뛴다. 원인 미확인([`v3d_perf_counters` 회차](v3d_perf_counters/runs/bootloader_20260925/summary.md)) |
| 표시 on 앱 내부 e2e p90 +1.9 ms | 디코더 1스레드에서 source pace 의 p90 만 3회 모두 높다(glass 평균은 같음). 원인 미확인([`dec_threads/`](dec_threads/summary.md) 결론 3) |
| 1스레드 디코드 처리량 상한 | 약 266 FPS 는 코드 주석(`ffmpeg_drm_decoder.cpp:180`) 기준이고 직접 재지 않았다. 다중 스트림·NPU 병목이 없는 용도에서는 자동이 나을 수 있다 |
| HEF 컨텍스트 조사 2단계 | 처리량을 실제로 움직일 지렛대는 config 20.5 MB 재적재와 컨텍스트 사이 텐서의 호스트 왕복(13.8 MB/프레임)이다([`candidate_reduction_survey/`](candidate_reduction_survey/summary.md) §5-3). 둘 다 "컨텍스트 수와 분할" 문제라 **컨텍스트가 적은 다른 분할의 HEF** 가 효과가 크지만 모델이 바뀐다 — **모델·HEF 교체는 사용자 결정**(재컴파일 DFC 는 x86 전용). 분석 도구는 저장소에 있다(`06_tools/python/hef_bytes.py`·`hef_byte_model.py`, 2026-10-11 회수) |
| 최소 지연 운용점(O2)의 표시 경로 | **일부 닫힘(2026-10-12)** — 페이싱 + 표시 on 을 구현·측정했다([`display_on`](input_pacing/runs/display_on/summary.md)): O2 Vulkan 합성 도착 → 화면 73.1 ms. **남은 것**: HVS 평면 b1/s1 교착(기존), 평면 b1/s2 의 vblank 3개 잠김, kms·표시 on 큐 ≥ 2 미측정, 실제 카메라. 아래는 측정 전 기록 — O2(`b1/s1 + --input-pace source --input-queue 1`)는 **표시 off 전용**이었다 — `--display on` 과 `--input-pace source` 를 함께 쓰면 종료 코드 2(표시 쪽 `pace_deadline` 이 이중으로 페이싱하고 평면 백엔드의 늦은 프레임 통계가 버림을 오판한다). 그래서 O2 의 glass-to-glass 는 **미측정**이다. 실제 카메라(CSI)의 촬영 시각·ISP 지연도 포함되지 않았다(페이싱은 파일 입력의 모사) — [`input_pacing/`](input_pacing/summary.md) §4 |
| 페이싱 b2 의 `e2e_arr` 꼬리 | **닫혔다(2026-10-12)** — 프레임별 로그([`frame_log`](input_pacing/runs/frame_log/summary.md))로 보니 최댓값은 실행 끝에 남은 프레임을 비우는 과정의 관찰 지연이었고, 정상 상태 최대는 약 107 ms(짝의 첫째가 33 ms 기다림). 아래는 측정 전 기록 — b2/s3·b2/s4 페이싱의 도착 기준 max 136~142 ms 가 p90 104 ms 보다 훨씬 크다. 시작 직후 프레임인지 드문 꼬리인지 프레임별 분포를 보지 않았다(프레임별 시각을 남기는 출력이 없다) — [`input_pacing/`](input_pacing/summary.md) §4 |
| 최소 지연 운용점의 열 조건 | O2 측정은 냉각 게이트 사이 30 s 실행이다. b1 은 처리량이 NPU 클럭에 비례하므로 칩이 데워져 클럭이 내려가면(약 104 °C 부터, [`thermal_soak/`](thermal_soak/summary.md)) 버림률이 18.6% 보다 늘 수 있다. 연속 운용은 재지 않았다 |
