---
name: verify
description: >
  이 프로젝트의 코드 변경을 검증하는 표준 절차. C/C++·GLSL 셰이더·CMake를 고치고 나서,
  또는 서브에이전트가 작업 결과를 보고하기 전에, 또는 07_results/에 결과를 쓰기 전에 쓴다.
  빌드 경고 0 → ctest → Vulkan 검증 레이어 → ASan/UBSan → perf → hailort.log 순서와
  각 단계의 합격 기준, 그리고 무시해도 되는 알려진 경고 목록을 담는다.
  트리거: "검증해", "테스트 돌려", "빌드·테스트 확인해줘", "빌드 확인", "ctest", "검증 레이어",
  "ASan", "sanitizer", "누수 확인", "perf", "보고 전 확인".
---

# 변경 검증 절차

작업 규모에 맞게 고른다. 한 줄 수정이면 1~2만, 셰이더·동기화·메모리를 건드렸으면 전부 돈다.
**각 단계의 실제 수치를 보고에 넣는다. "통과"라고만 쓰지 않는다.**

NPU를 쓰는 단계(2)는 Hailo-8 하나를 독점한다. 벤치마크가 돌고 있으면 `npu_guard` 훅이 막는다.

## 1. 빌드 — 경고 0

```bash
cmake -S . -B build -G Ninja && cmake --build build
```

- 합격: 경고·에러 0줄. `yvo_warnings` 인터페이스 타깃이 모든 타깃에 붙어 있다.
- 셰이더는 `yvo_add_shaders()`가 `glslc --target-env=vulkan1.3 -O -Werror`와
  `spirv-val --target-env vulkan1.3`을 자동으로 돌린다. 따로 부를 필요가 없다.
- 경고가 정말 0인지 보려면 `cmake --build build --clean-first 2>&1 | grep -ciE "warning|error"`.

## 2. 테스트 — ctest

```bash
ctest --test-dir build --output-on-failure
```

- `test_preproc` — GPU 전처리를 CPU 기준 구현(`01_src/ref/cpu_letterbox.hpp`)과 비교한다.
  합격 기준은 **max 오차 0**(근사가 아니라 비트 단위 일치)이다.
- `test_detections` — CPU / GPU ptr / GPU dmabuf / GPU dmabuf NHCW 경로의 검출을 비교한다.
  NPU를 독점하므로 `RUN_SERIAL`이고 느리다(약 19초). 합격 기준은 **매칭 비율 1.0000**이다.
- 전체 개수는 **195**이다(2026-10-27 — 평면 경로 계측 ee445a3의 `CommitMargin.*` 4개·`FrameLog` 1개, 기본 운용점 O2의 `DefaultSlots.*` 5개·`DisplayOptions.PlaneBackendOnlyWhenNamedExplicitly` 1개로 184 → 195, NPU 없이 192. 알려진 환경 실패: `04_data/raw_tensors/` 가 없으면 `PostprocFixture.*` 6개, `/dev/media0` 가 디코더로 잡히면 `IspPreprocTest.*` 10개. 그 앞 2026-10-16 — 메인 스레드 이벤트 대기·평면 슬롯 하한의 `SlotExchange.Wake*`·`*AcquirePoll`·`WaitEventUntil*` 5개와 `SlotBudget.PlanesMatchedNeedTwoSlots`·`OneSlotIsFineWithLatestOrComposite` 2개로 177 → 184, NPU 없이 181. 그 앞 2026-10-12 — 표시 페이싱(11040f3)의 `LateGapThreshold.*` 3개·`SlotBudget` 1개·`ResultCsv.Csv5*` 2개·`FrameLog` 1개로 170 → 177, NPU 없이 174. 그 앞 2026-10-12 — `FrameLog.*` 5개(`--frame-log` 서식)·`CliParse.*` 5개(`parse_stof`·`parse_size`·`arg_stof`)로 160 → 170, NPU 없이 167. 그 앞 2026-10-12 — 입력 페이싱 순수 로직·RESULT_CSV4 테스트 13개 추가로 147 → 160. 2026-10-11 — 하드웨어 없는 `test_pipeline_logic`(SlotExchange·SlotBudget·ResultCsv·CliParse)·`test_display_options` 추가로 95 → 147. NPU가 필요한 3개(`test_detections`·`test_postproc_live`·`test_gst_detections`)는 `npu` 라벨이 있어 `ctest -LE npu` 로 뺄 수 있다. 같은 날 앞서 `PostprocFixture.ImportHeapMatchesPlainBitExact` 추가: heap 입력 경로가 Plain과 비트 일치, NPU 불필요, dma-heap 노드가 없으면 skip. 이전 2026-09-29, `test_preproc` 의 `SandRefVsFfmpeg` 5개 — ref de-SAND 대 rpi-ffmpeg 바이트 일치; `test_gst_adapter` 10개 — HW HEVC 디코더와 GPU 필요, 없으면 skip; `test_gst_detections` 1개 — NPU 필요, RUN_SERIAL). `test_overlay`의 `KmsScanout.*`는 `/dev/dri/card1`을 열지만 **DRM master를
  가져가지 않는다** — 다른 프로세스(labwc)가 master일 때만 돌고, 아무도 master가 아니면 skip한다.

## 3. Vulkan 검증 레이어 — errors 0

```bash
VK_LOADER_LAYERS_ENABLE='*validation' build/bin/yvo_pipeline \
    --video 04_data/videos/video_h265.mp4 --mode dmabuf --layout nhcw --frames 200
VK_LOADER_LAYERS_ENABLE='*validation' build/bin/yvo_pipeline \
    --video 04_data/videos/video_h265.mp4 --mode dmabuf --layout nhcw --frames 200 --input-pace asap --batch 8
```

- 2026-10-27부터 기본값이 O2(`--batch 1 --slots 1 --input-pace source`)라 첫 명령은 페이싱 루프(`run_paced`, 약 7 s)를, 둘째 명령은 asap 루프(배치 8)를 덮는다. **둘 다 돌린다.**
- 합격: 각 출력 끝 줄의 `vk validation errors 0`. 2026-10-08부터 `--postproc` 기본값이 gpu 라 이 명령은 GPU 후처리(`yolo_decode`·`efficient_nms`)까지 덮는다.
- 셰이더·배리어·동기화·dma-buf import를 건드렸으면 반드시 돈다.
- `yvo_preproc_bench`도 같은 방식으로 돌릴 수 있다(NPU를 쓰지 않아 더 빠르다).

### 3-1. 표시 경로를 건드렸을 때

`01_src/display/`를 바꿨으면 해당 백엔드로 실제 표시를 한 번 돌린다(`--batch 2 --slots 4 --input-pace asap --postproc gpu --display on`. 평면 백엔드는 `--slots 6`).
**항상 `timeout -s KILL 90`으로 감싼다.**
- 먼저 `XDG_RUNTIME_DIR=/run/user/1000 WAYLAND_DISPLAY=wayland-0 wlr-randr`에 **HDMI-A-1**이 있는지 본다.
  `Headless output`만 있으면 labwc가 HDMI를 잃은 것이다. 이 상태에서 스왑체인 경로(`wayland`)는 `vkQueuePresentKHR`에서
  무기한 멈추고 SIGTERM도 안 먹는다. `/sys/kernel/debug/dri/1/clients`의 master 열을 보고 복구한 뒤 돌린다.
- `wayland-planes`/`kms`: 요약의 `scanout zero_copy 1, color_ok 1, hvs_underrun delta 0`이 합격 기준이다.
  debugfs 줄(`video=plane[83] NV12/SAND128 … boxes=plane[107] AB24`)을 보고에 넣는다.
- `kms`는 DRM master가 필요하다. 화면이 사용자 눈앞에서 바뀌므로 **사용자가 자리에 있는지 먼저 확인받고** 돌린다
  (`CLAUDE.md` §사용자 확인). 확인 전 기본은 건너뛰기다. `loginctl activate <tty 세션 id>`로 데스크톱을 비활성화하고
  실행한 뒤 `loginctl activate <데스크톱 세션 id>`로 돌아간다. **실행 중에는 VT를 바꾸지 않는다.**
  요약의 `kms: flips = presented, flip_timeouts 0, master_lost no`를 확인한다.
  종료 복원이 `restore failed (EINVAL) -> planes-off ok`로 나오는 것은 지금 알려진 동작이다.

## 4. ASan/UBSan — 에러·누수 0

**반드시 별도 빌드 디렉터리를 쓴다.** `build/`를 새니타이저로 덮으면 측정이 무의미해진다.

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DYVO_SANITIZE=ON
cmake --build build-asan
build-asan/bin/yvo_pipeline --video 04_data/videos/video_h265.mp4 \
    --mode dmabuf --layout nhcw --frames 100
build-asan/bin/yvo_pipeline --video 04_data/videos/video_h265.mp4 \
    --mode dmabuf --layout nhcw --frames 100 --input-pace asap --batch 8
```

- 2026-10-27부터 기본값(O2)은 페이싱 루프만 지나므로 asap 루프도 한 번 더 돌린다(§3과 같은 이유).

- `YVO_SANITIZE=ON`(2026-10-11)은 C 스텁까지 모든 타깃의 컴파일·링크에 `-fsanitize=address,undefined`를 넣는다. 예전 방식(`CMAKE_CXX_FLAGS`)으로 설정된 `build-asan/` 캐시가 남아 있어도 플래그가 겹칠 뿐 해롭지 않다. TSan은 이 보드에서 돌지 않는다(47비트 VMA 미지원).
- 합격: ASan/UBSan 출력 없음, `LeakSanitizer` 누수 0. (위 명령은 기본값인 GPU 후처리 경로를 돈다. host NMS 경로를 보려면 `--postproc host`.)
- 디코더·dma-buf 수명·슬롯 반납을 건드렸으면 반드시 돈다.
  (과거에 `avcodec_flush_buffers()` 누락으로 1448 B 누수가 이렇게 잡혔다.)
- fd 누수는 `yvo_drm_dump`가 시작·종료 fd 수를 세어 확인한다.

## 5. perf — CPU 사용처 확인

```bash
perf record -q -g --call-graph=fp -o perf.data -- <명령>
perf report -i perf.data --no-children --stdio -g none --percent-limit 0.2
```

- CPU 시간이 예상과 다르면 돈다. `kernel.perf_event_paranoid=1`이라 일반 사용자도 커널 구간까지 본다.
- `libhailort`는 심볼이 없어 주소로 나온다. 알려진 주소:
  - `0x1c10cc` 부근 — `InputTransformContext::transform` (NHWC→NHCW). **NHCW 경로에서는 0이어야 한다.**
  - `0x30ffac` 부근 — NMS·후처리로 추정(미확정).
- 전체 CPU 양은 `Event count (approx.)`로 비교한다(비율보다 이쪽이 정직하다).

## 6. hailort.log — error/warning 0

```bash
grep -cE '\[error\]|\[warning\]' hailort.log
```

- `hailort.log`는 **실행한 디렉터리**에 생긴다. 보관 위치는 `STRUCTURE.md` §3.5 — 검증 로그로 남길 것은 `07_results/<실험>/verification/hailort_<무엇>.log`, 그 밖의 디버깅 로그는 `11_logs/<날짜>_<무엇>.log`.
- 온도·클럭 경고가 있으면 그 실행의 **성능 수치는 신뢰할 수 없다**. `measure` 스킬을 볼 것.

---

## 무시해도 되는 알려진 경고

아래는 전부 정상이다. **"남은 이슈"로 보고하지 않는다.**

| 메시지 | 이유 |
|---|---|
| `[vulkan WARNING] Unable to open device : ... VK_ERROR_INCOMPATIBLE_DRIVER` | V3D가 아닌 장치를 훑는 과정이다. V3D가 선택되면 정상이다 |
| 로더의 `Layer forced enabled due to env var` (2건) | `VK_LOADER_LAYERS_ENABLE`을 쓰면 항상 나온다 |
| hailort `[info] ... HAILO_SHUTDOWN_EVENT_SIGNALED(57)` | 정상 종료 메시지다(`[info]` 레벨) |
| `vcgencmd get_throttled` = `0x80000` | 과거 기록 비트다. 현재 스로틀링이 아니다(현재 비트는 하위 4비트) |

진짜로 봐야 하는 것: 검증 레이어 **errors**, ASan/UBSan 리포트, hailort `[error]`,
hailort `[warning]` 중 온도·클럭 관련.

## 로그 보관

검증 로그는 `07_results/<실험>/verification/`에 남긴다. 이름은 내용을 드러내게 쓴다
(예: `validation_dmabuf_200f.log`, `asan_ptr_100f.log`, `perf_nhcw_b8_top.txt`).
