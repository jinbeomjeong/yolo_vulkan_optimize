---
name: verify
description: >
  이 프로젝트의 코드 변경을 검증하는 표준 절차. C/C++·GLSL 셰이더·CMake를 고치고 나서,
  또는 서브에이전트가 작업 결과를 보고하기 전에, 또는 07_results/에 결과를 쓰기 전에 쓴다.
  빌드 경고 0 → ctest → Vulkan 검증 레이어 → ASan/UBSan → perf → hailort.log 순서와
  각 단계의 합격 기준, 그리고 무시해도 되는 알려진 경고 목록을 담는다.
  트리거: "검증해", "테스트 돌려", "확인해줘", "빌드 확인", "ctest", "검증 레이어",
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

- `test_preproc` — GPU 전처리를 CPU 기준 구현(`01_src/common/cpu_letterbox.hpp`)과 비교한다.
  합격 기준은 **max 오차 0**(근사가 아니라 비트 단위 일치)이다.
- `test_detections` — CPU / GPU ptr / GPU dmabuf / GPU dmabuf NHCW 경로의 검출을 비교한다.
  NPU를 독점하므로 `RUN_SERIAL`이고 느리다(약 19초). 합격 기준은 **매칭 비율 1.0000**이다.

## 3. Vulkan 검증 레이어 — errors 0

```bash
VK_LOADER_LAYERS_ENABLE='*validation' build/01_src/pipeline/yvo_pipeline \
    --video 04_data/videos/video_h265.mp4 --mode dmabuf --layout nhcw --frames 200
```

- 합격: 출력 끝 줄의 `vk validation errors 0`.
- 셰이더·배리어·동기화·dma-buf import를 건드렸으면 반드시 돈다.
- `yvo_preproc_bench`도 같은 방식으로 돌릴 수 있다(NPU를 쓰지 않아 더 빠르다).

## 4. ASan/UBSan — 에러·누수 0

**반드시 별도 빌드 디렉터리를 쓴다.** `build/`를 새니타이저로 덮으면 측정이 무의미해진다.

```bash
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan
build-asan/01_src/pipeline/yvo_pipeline --video 04_data/videos/video_h265.mp4 \
    --mode dmabuf --layout nhcw --frames 100
```

- 합격: ASan/UBSan 출력 없음, `LeakSanitizer` 누수 0.
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

- `hailort.log`는 **실행한 디렉터리**에 생긴다. 보관할 것은 `10_logs/`에 날짜를 붙여 옮긴다.
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
