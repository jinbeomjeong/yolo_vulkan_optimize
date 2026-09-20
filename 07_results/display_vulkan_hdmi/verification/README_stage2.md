# 표시 트랙 2단계 검증 기록 ([4] VulkanSwapchain / [5] display_convert.comp / dmabuf_export)

2026-09-20. `.claude/skills/verify/SKILL.md` 절차를 그대로 따랐다. 1단계 파일은 덮어쓰지 않고
`_stage2` 접미사로 새로 남겼다.

| # | 게이트 | 기준 | 결과 | 로그 |
|---|---|---|---|---|
| 1 | 빌드 경고 0 | `cmake --build build --clean-first` 경고·에러 **0줄** | **0** | `clean_build_stage2.log` |
| 2 | ctest | 11/11, `test_preproc` max 오차 0, `test_detections` 매칭 1.0000 | **11/11**, max_err=0 (5프레임 전부), ratio **1.0000** ×3경로, preproc bit-identical 100/100 | `ctest_11of11_stage2.log` |
| 3 | P6 회귀 | `yvo_preproc_bench --frames 600 --layout nhcw --repeat 3` GPU median ∈ [2.074, 2.162] ms | **2.122 / 2.127 / 2.136** — 전부 통과 (NPU 유휴 확인, 66.4 °C) | `preproc_bench_stage2.txt` |
| 4 | 검증 레이어 | `--probe present` / `--probe convert`, `--sync-validation`, `--display-blit`, 실행 중 리사이즈 전부 errors 0 | **errors 0 / warnings 0** (6개 구성 전부) | `validation_p05_matrix.log`, `validation_p08_convert.log`, `validation_p08_convert_blit.log` |
| 5 | ASan/UBSan | 별도 `build-asan/`, 에러·누수 0, fd 누수 0 | **ASan/UBSan 리포트 0줄, LeakSanitizer 누수 0**, fd 정상 상태 유지 | `asan_stage2_probes.log`, `asan_stage2_convert.log`, `fdcount_convert_200f.txt` |
| 6 | 프로브 | P4 / P5 / P8 합격 기준 | 전부 PASS (수치는 `../evidence/p04_p05_p08_probe.md`) | `probe_p04_surface.log`, `probe_export_dmabuf.log`, `convert_workgroup_sweep.log` |

## 1. 빌드

```
$ cmake --build build --clean-first 2>&1 | grep -ciE "warning|error"
0
```
`build-asan` 도 `yvo_display_probe` 타깃까지 경고 0.

## 2. ctest

```
100% tests passed, 0 tests failed out of 11
Total Test time (real) =  70.29 sec
```
```
frame    0 … max_err=0 mean_err=0.00000 hist[0]=691200 [1]=0 [2]=0 [>=3]=0 pad_mismatch=0
frame  100 … max_err=0 …   frame  500 … max_err=0 …
frame  900 … max_err=0 …   frame 1815 … max_err=0 …
[gpu_ptr]         frames 100, detections cpu 203 / gpu_ptr 203, matched 203, ratio 1.0000, preproc bit-identical 100/100
[gpu_dmabuf]      frames 100, detections cpu 203 / gpu_dmabuf 203, matched 203, ratio 1.0000, preproc bit-identical 100/100
[gpu_dmabuf_nhcw] frames 100, detections cpu 203 / gpu_dmabuf_nhcw 203, matched 203, ratio 1.0000, preproc bit-identical 100/100
```

## 3. P6 회귀 (전처리 2.13 ms 경로가 그대로인가)

측정 전 `ps -eo comm | grep yvo_` 0건(NPU 유휴), `vcgencmd measure_temp` 66.4 °C.

```
[decode+gpu] gpu shader ms: median=2.122 p90=2.233 mean=2.141 min=2.048 max=2.784 (n=600)
[decode+gpu] gpu shader ms: median=2.127 p90=2.188 mean=2.141 min=2.055 max=2.758 (n=600)
[decode+gpu] gpu shader ms: median=2.136 p90=2.311 mean=2.177 min=2.040 max=2.860 (n=600)
```
게이트 [2.074, 2.162] 안. `SandLetterbox`·`sand_letterbox.comp` 는 한 줄도 건드리지 않았고
`yvo_display` 는 별도 라이브러리라 `yvo_preproc_bench` 의 링크 그래프도 바뀌지 않는다.

## 4. Vulkan 검증 레이어 (errors 0)

| 구성 | 명령 | errors / warnings |
|---|---|---|
| P5 fullscreen, storage 직접 쓰기 | `--probe present --display-fullscreen --sync-validation` | **0 / 0** |
| P5 fullscreen, blit 폴백 | `+ --display-blit` | **0 / 0** |
| P5 창 1280x720, storage, 실행 중 리사이즈 | `--display-size 1280x720 --resize-every 20 --sync-validation` | **0 / 0** |
| P5 창 1280x720, blit, 실행 중 리사이즈 | `+ --display-blit` | **0 / 0** |
| P8 fullscreen, storage | `--probe convert --frames 300 --display-fullscreen --sync-validation` | **0 / 0** |
| P8 fullscreen, blit | `+ --display-blit` | **0 / 0** |

`--resize-every 20` 은 실행 중 `xdg_toplevel.set_fullscreen` 을 토글해 **컴포지터가 실제로 새 크기를
configure** 하게 만든다 → 스왑체인 재생성 5회, 구 스왑체인 폐기, 세마포어 수명 전환을 모두 지난다.
매 실행 1회는 acquire 후 일부러 그리지 않고 `vkReleaseSwapchainImagesEXT` 로 되돌린다.

**이 단계에서 검증 레이어가 잡아낸 실제 버그 3개는 모두 수정했다** — 상세는
`../evidence/p04_p05_p08_probe.md` 의 P4/P5 절. 요약:
1. `minImageCount` 3 (FIFO 모드별 값)이 `VUID-...-presentMode-02839` 위반 → 기본을 4로 바꾸고
   3은 `--lean-images` opt-in 으로 남김
2. blit 폴백에서 자체 이미지 레이아웃 전환이 이전 프레임의 `vkCmdBlitImage` 읽기와 WRITE_AFTER_READ
   → 배리어 `srcStageMask` 에 `BLIT|COPY` 추가
3. `vkReleaseSwapchainImagesEXT` 를 세마포어 소비 제출이 끝나기 전에 호출
   → 펜스 신호 후로 미루는 `flush_pending_releases()` 추가

## 5. ASan / UBSan (별도 `build-asan/`)

```
cmake -S . -B build-asan -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
cmake --build build-asan --target yvo_display_probe
ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1
```

돌린 구성: `--probe surface`, `--probe export`, `--probe present`(창+리사이즈 토글 / fullscreen+blit),
`--probe convert --frames 60 --display-fullscreen`.

```
$ grep -cE "ERROR: (AddressSanitizer|LeakSanitizer)|runtime error" asan_stage2_probes.log asan_stage2_convert.log
0
0
```
ASan 아래에서도 결과가 같다: 픽셀 비트 일치(max 오차 0), 1080p shader median 4.244 ms,
present 60/60, 검증 레이어 errors 0.

### fd 누수

`--probe convert --frames 200 --display-fullscreen` 실행 중 0.4 s 간격으로 `/proc/<pid>/fd` 를 셌다
(`fdcount_convert_200f.txt`):

```
3 63 63 63 63 63 63 63 63 63 63 67 67 67 67 67 67 67 67
```
- 오프스크린 3개 크기(1920x1080 / 1600x900 / 1280x720)를 연달아 돌면서 **63에서 전혀 늘지 않았다** —
  각 구간이 디코더와 `DisplayConvert`(따라서 `DmaBufImportCache` 의 dup fd 전부)를 새로 만들고 닫는다.
- 스왑체인 구간에서 +4 (Wayland 소켓 등) 후 200프레임 동안 67 유지.
**단조 증가가 없다 → fd 누수 없음.**

## 6. 알려진 무해 경고

- `[vulkan WARNING] Unable to open device : ... VK_ERROR_INCOMPATIBLE_DRIVER` — V3D 가 아닌 장치를 훑는 과정
  (검증 레이어를 명시적으로 켠 실행에서는 이 경고도 사라져 warnings 0 이 된다)
- `Layer "VK_LAYER_KHRONOS_validation" forced enabled due to env var` — `VK_LOADER_LAYERS_ENABLE` 사용 시 항상
- `vcgencmd get_throttled` = `0x80000` — 과거 기록 비트
