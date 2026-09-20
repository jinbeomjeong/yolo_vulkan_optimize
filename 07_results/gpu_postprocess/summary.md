# NPU 원시 출력 → GPU 후처리 — 프로브 결과 (P1 / P2)

- 측정일: 2026-09-20
- 환경: Raspberry Pi CM5 (BCM2712), Hailo-8 **PCIe 8.0 GT/s x1 = Gen3**(sysfs 실측), HailoRT 4.24.0,
  V3D 7.1.10.2 / V3DV Mesa 26.2.2, kernel 6.18.50, g++ 14.2 Release
- 모델: `03_models/yolov11m.hef` / 영상: `04_data/videos/video_h265.mp4` (1080p30)
- 설정: batch 8, slots 16, 입력 640×640 NHCW. 반복은 항목별로 아래에 표기
- 목적: 계획서(`/home/pi/.claude/plans/jaunty-dazzling-quasar.md`) §4의 GPU 후처리 설계에서
  **본구현 전에 닫아야 할 가정**을 검증한다. `07_results/nms_host_cost/`가 타당성을 판정한 다음 단계다.

## 결과

| 프로브 | 기준 | 실측 | 판정 |
|---|---|---|---|
| **P1 ①** 원시 출력 6개, 이름·shape | 계획서 §4.0 표와 일치 | 일치. `hw_shape`에만 P5 패딩(20×**24**), `shape`는 20×20 | **통과** |
| **P1 ②** `get_quant_infos().size() == 1` | 스칼라 양자화 | 6스트림 전부 `.size()==1`, 값도 표와 동일 | **통과** (LUT 버퍼 불필요) |
| **P1 ③** API 최소 하나 동작 | 7콜백 전부 SUCCESS | **`infer_async`·`read_async` 둘 다** 각 56콜백 실패 0. 두 경로 텐서가 **40/40 프레임 바이트 일치** | **통과** |
| **P1 ④** CPU 기준 ↔ HailoRT NMS | `match()` ratio ≥ 0.95 | **1.0000** (40프레임, ref 81 / test 81 / matched 81) | **통과** |
| **P1 ⑤** P5 패딩 열 제외 | 제외가 맞다 | 제외·포함이 같은 결과. **증거가 약하다**(아래 주의) | 통과(약) |
| **P1 ⑥** 처리량 | FPS ≥ 50, log error 0 | **51.90 FPS**(200프레임 ×3 중앙값, min 51.89/max 51.92). `hailortcli benchmark` 기준선 51.00 대비 **+1.8%** | **통과** |
| **P2 ①②③** Vulkan dma-buf D2H | 매핑·콜백·바이트 일치 | `HAILO_SUCCESS`, 콜백 210개 실패 0, malloc 대비 **30/30 프레임 완전 일치** | **통과** |
| **P2 ④** size 조합 | 최소 하나 동작 | 16셀 전수 → **8/16 통과, 전부 `read=hw_frame_size`** | **통과** |
| **P2 ⑤** 매핑 96회 < 100 ms | – | **0.261 ms** (0.003 ms/회). 378배 여유 | **통과** |
| **P2 ⑥** `cpu_read_1.22MB_ms` | 기록 | malloc 0.244 / heap 0.396 / **vulkan 0.826 ms** | 기록 |
| **P3 ①** 픽스처 G5 | 후보 수 동일, 점수 비트 일치, 좌표 < 0.25 px | 5프레임 **cpu 115 / gpu 115 완전 일치**, onlyCPU·onlyGPU **0**, 점수 비트 불일치 **0**, 좌표 최대 **0.000122 px**(기준의 **1/2000**) | **통과** |
| **P3 G6** 결정성 | 10회 반복 동일 | `(score, anchor_key)` 전순서 정렬 후 **전부 바이트 동일** | **통과** |
| **P3 ②** live 바이트 합 | 셰이더 == CPU | 6텐서 × 20프레임 **불일치 0**. 디코드도 20프레임 전부 개수·점수 비트 일치 | **통과** |
| **P3 ③** 스테일(배리어 ON) | A/B 각각 정확 | A 25/25, B 10/10, sum_mismatch 0 | **통과** |
| **P3 ④** 배리어 OFF | 기록 | **틀린 결과가 나오지 않았다** — 아래 주의 참조 | 기록 |
| **P3 ⑤** 검증 레이어 | errors 0 | fixture·live·stale 전부 **errors 0**(`--sync-validation` 포함) | **통과** |
| **P3 ⑥** pass 1 GPU ms | §4.7 추정 대조 | **median 0.562~0.569 ms** (p90 0.629~0.637) — 추정 **0.4~0.9 ms 안** | **통과** |

## 결론

1. **원시 출력 경로가 우리 코드로 동작한다. HEF 재컴파일이 필요 없다.**
   `VDevice::configure()` → `ConfiguredNetworkGroup::infer_async()`(채택) 또는
   `get_output_streams()+read_async()`(대체) 양쪽이 모두 성립하고 결과가 바이트 단위로 같다.
   **`InferModel`로는 불가능함이 확정됐다** — `set_format_order(NHCW)` 후 `configure()`가
   `HAILO_INVALID_ARGUMENT`로 실패하고 `nms_post_process.cpp:42`가 원인을 찍는다.
2. **텐서 해석 4개 미확정 변수가 전부 확정됐다**(40프레임 16조합 전수 시도):
   | 변수 | 확정값 | 판별력 |
   |---|---|---|
   | DFL 채널 순서 | **`ch = coord*reg_max + bin`** | 반대면 ratio 0.0000 (박스 81 → 462개 폭발). 가장 강하게 갈린다 |
   | sigmoid | **적용하지 않는다** | 재적용하면 후보 15.1 → 16800/frame |
   | 앵커당 클래스 | **다중 클래스** | argmax 만 쓰면 81개 중 1개를 놓쳐 0.9877. **2개 이상이 0.2를 넘는 앵커가 실재한다** |
   | threshold 비교 | `>=` | **원리적으로 구분 불가**: `0.2/qp_scale`이 어느 스케일에서도 정수가 아니다(57.6409/53.1098/52.3728) |
3. **클래스 출력이 이미 확률이므로 정수 도메인 사전필터가 가능하다.**
   `q_thr_low = floor(0.2/qp_scale)` = P3 **57** / P4 **53** / P5 **52**.
   672,000회의 float 곱·비교가 바이트 비교로 바뀐다 — pass 1 비용의 대부분이 여기서 사라진다.
4. **§4.2 채택안(Vulkan 할당 → export → Hailo `dma_map_dmabuf` D2H)을 바꿀 이유가 없다.**
   `HOST_COHERENT` 때문에 CPU 읽기가 heap 의 2.1배(0.826 vs 0.396 ms)지만,
   §4.4 하이브리드안에서 CPU 가 읽는 것은 압축된 후보 ~1.6 KB뿐이라 무해하다.
5. **크기 계약 확정**: `dma_map_dmabuf`는 `hw_frame_size`·`dmabuf_size` 둘 다 받고,
   `read_async`/`hailo_dma_buffer_t.size`는 **`hw_frame_size` 뿐이다**
   (`core_op.cpp:524`가 거부). 매핑은 `dmabuf_size`를 쓴다(버퍼 전체를 덮는 쪽이 안전).
6. **원시 출력이 처리량을 떨어뜨리지 않는다.** 51.90 FPS 로 `hailortcli benchmark` 기준선 51.00 보다
   1.8% 높다. `07_results/nms_host_cost/`의 `raw_async 51.62 vs full_async 51.22`와 같은 방향이다.

7. **GPU pass 1 이 설계대로 성립한다.** 정수 사전필터 + NHCW coalescing + 스케일별 3 디스패치가
   **0.562 ms** 로 §4.7 추정 중앙에 들었다. 후보 개수·점수가 CPU 기준과 완전히 같고 좌표 잔차는
   `exp()` 구현 차이뿐이다(0.000122 px).
8. **하이브리드안(§4.4 A)의 호스트 비용이 추정보다 싸다.** 후보 read 1.1~4.0 µs +
   `cpu_nms_greedy_by_class` 1.6~6.6 µs = **3~11 µs** (계획서 추정 5~20 µs). 최종 검출 수도 기준과 일치.
   **완전 GPU NMS([H])를 서둘러야 할 이유가 없다.**
9. **워크그룹 규율이 전처리와 반대다.** `sand_letterbox` 는 "invocation 이 클수록 좋다"였는데
   (64→3.65, 256→2.13 ms) `yolo_decode` 는 **60 invocation(10×6) 0.566 ms 가 256 invocation(0.733~0.987)보다 빠르다**.
   x 를 키워 coalescing 을 늘리려는 시도도 역효과였다(80×3 = 1.269 ms).
   **32·64·128(16의 배수)이 20·40·60·120보다 느린 이유는 설명하지 못했다.**
   유효 invocation 이 2,100개뿐이라 큰 워크그룹에서는 워크그룹 수가 8개까지 떨어져 코어에 덜 퍼진다는 것이
   담당자 가설이나 **측정치만이 근거인 미확정 가설**이다.

## 주의와 남은 일

- **⚠ `infer_async`에 틀린 크기를 넘기면 `HAILO_SUCCESS`를 돌려주고 콜백을 하나도 호출하지 않는다.**
  요청이 내부 워커에서 조용히 버려져 **무한 대기하는 코드는 영구 정지한다**(P2 첫 실행이 실제로 멈췄고
  gdb 로 확인했다). `read_async`는 발행 시점에 `HAILO_STREAM_ABORT(63)`을 돌려주므로 관측 가능하다.
  → **러너 재작성(`hailo_raw_runner`)은 프레임 완료 대기에 반드시 시한을 두고, 시한 초과를
  치명적 오류로 올려 종료해야 한다.**
- **HailoRT `read_async`는 16384 B 정렬을 요구한다.** 4096 B 로 주면 프레임×스트림마다
  `vdma_stream.cpp:379` 경고가 뜬다(첫 실행 1,477건). `DmaBufExporter`의 `align` 파라미터로 처리했고
  최종 실행의 `unaligned buffer` 경고는 **0건**이다.
- **P1 ⑤는 약한 증거다.** 이 영상에서 **P5 클래스 텐서의 패딩 열이 전부 0**이라(box 텐서만 max q=147)
  포함/제외가 같은 결과를 냈다. 그래도 셰이더는 `w < 20`으로 순회해야 한다 —
  `w=20..23`은 `(w+0.5)*32 ≥ 656`으로 640 입력 밖 좌표를 만들고 패딩 값은 정의되지 않았다.
- **Vulkan dma-buf 는 V3DV 의 "+256 B" + 16 KB 올림 때문에 텐서당 16~29 KB 를 더 쓴다.**
  슬롯 16이면 19.5 MB → 약 **21.8 MB**(+2.3 MB). 문제 없는 크기다.
- **프레임 지연은 이 프로브의 개선 대상이 아니다.** slots 16이면 중앙값 142 ms / p90 268 ms 다.
  계획서 §5.3의 `slots + gpu_depth ≤ 10` 제약과 P9의 b4/s4 측정이 이 값을 정한다.
- **⚠ P3 ④가 "배리어 없이도 맞음"으로 나왔다.** 배리어 필수성을 실증으로 보이려던 증거는 확보하지 못했다.
  **이것은 배리어가 불필요하다는 증거가 아니다** — 큐 패밀리 소유권 전환은 스펙 요구사항이고
  (VUID-VkBufferMemoryBarrier2-buffer-09095/09096), V3D 의 유일한 메모리 타입이 `HOST_COHERENT` 라
  이 하드웨어에 스테일 경로가 없는 것으로 **추정**할 뿐이다. 배리어는 유지한다.
- **`--live`/`--stale` 은 batch 1 순차 실행이다.** NPU 파이프라이닝과 함께 돌릴 때의 처리량·지연,
  GPU↔NPU 메모리 대역폭 경합(부록 A.8)은 측정하지 않았다. [F] 범위다.
- (완료) ~~**GPU 셰이더(`yolo_decode.comp`)와 P3 는 아직 하지 않았다.**~~ 이 프로브는 dma-buf 경로와
  텐서 해석까지만 닫는다. 픽스처(`04_data/raw_tensors/`, 5프레임 × 6텐서 = 5.9 MB + `fixtures.json`)는
  **NPU 없이 셰이더를 디버깅하도록** 준비돼 있다.
- **처리량과 지연은 개선되지 않는다**(계획서 §4.8). 51.63 FPS 에서 51.6 이고, 호스트 NMS 2~3 ms 가
  사라지는 대신 submit+펜스 1회(+1~2 ms)가 붙는다. 얻는 것은 **CPU 코어 약 0.16개**와,
  입력 해상도·스트림 수를 늘릴 때 선형으로 커질 고정 비용을 미리 없애는 것이다.

## 파일

- `evidence/stream_infos_quant.txt` — 6스트림 shape·format·`get_quant_infos()` 전문
- `evidence/p1_combination_sweep.txt` — 텐서 해석 16조합 전수 시도 표
- `evidence/infer_model_raw_attempt.txt` — `InferModel` 경로 폐기 증거
- `evidence/p1_api_result.txt`, `p1_stdout.log`, `hailort.log`
- `evidence/hailortcli_benchmark_b8_baseline.txt` — 같은 날 냉각 후 기준선(51.00 FPS)
- `evidence/p2_alloc_matrix.txt` — size 4조합 × 2 alloc × 2 API = 16셀
- `evidence/p2_result.txt`, `p2_cpu_read.txt`, `p2_stdout.log`, `p2_asan.log`, `hailort_p2.log`
- `evidence/p3_{fixture,sweep,live,stale}.txt` — G5·G6·바이트 합·스테일·워크그룹 26조합
- `verification/ctest_11of11.log`, `ctest_golden_numbers.log`, `asan_ubsan_raw_probe.log`
- `04_data/raw_tensors/` — 셰이더 디버깅용 픽스처 (`.gitignore` 에 있음)
