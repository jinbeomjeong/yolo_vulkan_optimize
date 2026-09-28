# ffmpeg CLI 대조군의 출력 포맷 비교 (미지정 / DRM_PRIME 유지 / hwdownload NV12·YUV420P / vf_unsand)

## 측정 조건
- 측정 일시: 2026-09-29 03:14~03:20 KST
- 환경: 부트로더 2026-09-25 적용 후, kernel 6.18.50, rpi-ffmpeg `7.1.5-0+deb13u1+rpt2`, governor ondemand, `throttled=0x0`. 상세는 [`env_bench.txt`](env_bench.txt).
- 배경
  - 부모 측정([`../../summary.md`](../../summary.md))의 대조군 `h265_Cref_ffmpeg_drm_S1`(`ffmpeg -hwaccel drm -i … -f null -`, 332.8 FPS / CPU 2.0 ms/f)이 "HW 디코드 CPU 비용"으로 인용돼 왔다.
  - 그러나 출력 포맷을 지정하지 않으면 프레임을 CPU로 내려받는지가 확인되지 않은 상태였다.
  - 같은 날 GStreamer rpt4의 DMABuf 경로가 0.24 ms/f로 측정돼 차이의 원인을 가려야 했다([`../gst_rpt4/summary.md`](../gst_rpt4/summary.md)).
- 명령: `ONLY='^h265_Cref_ffmpeg_(drm_S1|outfmt_(drmprime|dl_nv12|dl_yuv420p)_S1)$' RUNS=3 OUT_DIR=… bash 06_tools/bench/decode_bench.sh`
  - `outfmt` 조합은 이 측정을 위해 스크립트에 넣은 선택 조합(opt-in)이다.
  - 방법은 부모와 같다: wait4 rusage, 조합마다 3회, 중앙값, 65 °C 냉각.
- 입력: `04_data/videos/video_h265.mp4`(HEVC Main 1080p30, 1816프레임). 12회 모두 1816프레임을 끝까지 디코드했다.

| id | ffmpeg 인자 (`-hide_banner -nostdin -hwaccel drm` 뒤) |
|---|---|
| `drm_S1` (기준선, 부모와 같은 명령) | `-i V -f null -` |
| `outfmt_drmprime_S1` | `-hwaccel_output_format drm_prime -i V -f null -` |
| `outfmt_dl_nv12_S1` | `-hwaccel_output_format drm_prime -i V -vf hwdownload,format=nv12 -f null -` |
| `outfmt_dl_yuv420p_S1` | `-hwaccel_output_format drm_prime -i V -vf hwdownload,format=yuv420p -f null -` |
| (`outfmt_unsand_S1`) | `… -vf hwdownload,format=sand128,unsand,format=yuv420p` → **실행 불가**(아래 주의) |

## 결과 (중앙값, 3회)

CPU ms/frame = 중앙값 실행의 (user+sys) / 1816.

| 조합 | FPS | FPS 범위 | CPU ms/frame | avg cores | max RSS MB |
|---|---|---|---|---|---|
| `drm_S1` (미지정) | 334.34 | 321.42~334.61 | **2.02** | 0.677 | 92.9 |
| `outfmt_drmprime_S1` | **386.85** | 385.88~387.83 | **0.55** | 0.212 | 84.8 |
| `outfmt_dl_nv12_S1` | 336.51 | 336.21~336.55 | 2.14 | 0.721 | 91.3 |
| `outfmt_dl_yuv420p_S1` | 332.76 | 332.12~333.97 | 2.21 | 0.735 | 91.3 |

perf(1회, 콜 그래프 없음, [`evidence/perf_*_top.txt`](evidence/)):

| 조합 | Event count (cycles) | 상위 항목 |
|---|---|---|
| `drm_S1` | 8.97 G | `libavutil` 0x2115c~0x21228 합계 약 51%, 커널 `__pi_dcache_inval_poc` 6.7% |
| `outfmt_dl_nv12_S1` | 9.63 G | `libavutil` 0x2115c~0x21168 합계 약 52%, `__pi_dcache_inval_poc` 6.9% |
| `outfmt_drmprime_S1` | 2.41 G | `memset` 12.6%, 동적 링커, 커널 복사 등 CLI 자체 비용. `libavutil` 복사 루프 없음 |

- `libavutil` 0x2115c 부근은 내보내지 않은 함수다. 128바이트를 NEON `ld1`/`st1`로 옮기고 `add x13, x13, x10`으로 한 열 너비씩 건너뛰는 루프로, SAND 열을 평면 행으로 옮기는 복사 구조다([`evidence/libavutil_hot_disasm.txt`](evidence/libavutil_hot_disasm.txt)).
  - rpi-ffmpeg의 NEON SAND→planar 경로(`ff_rpi_sand8_lines_to_planar_*` 계열, 공개 래퍼 `av_rpi_sand_to_planar_*`)로 본다(**추정**: 심볼이 없어 구조로 판단).
- `__pi_dcache_inval_poc`는 CPU가 dma-buf를 읽기 전에 하는 캐시 무효화로 본다(**추정**).

## 결론
- **부모 대조군(`-hwaccel drm`, 출력 포맷 미지정)은 프레임을 CPU로 내려받아 `yuv420p`로 변환한다.**
  - `-v verbose`에서 필터 그래프 입력이 `pixfmt:yuv420p`이고, 디코더 로그는 `swfmt rpi4_8`(SAND), `dst DMABuf`다.
  - CPU 2.02 ms/f 중 약 1.5 ms/f가 이 내려받기와 SAND→planar 복사다. DRM_PRIME을 유지하면 0.55 ms/f로 줄고 cycles는 73% 줄어든다.
  - 그래서 "ffmpeg `-hwaccel drm` 332.8 FPS / 2.0 ms/f"는 **HW 디코드 자체의 비용이 아니라 CPU 내려받기가 들어간 값**이다. 부모의 설명("frames not downloaded to CPU")은 틀렸다. 스크립트 주석은 2026-09-29에 고쳤다.
- **복사 없는 경로끼리는 세 구현의 수치가 비슷하다.**

  | 경로 | FPS | CPU ms/f |
  |---|---|---|
  | ffmpeg CLI DRM_PRIME 유지 | 386.9 | 0.55 |
  | GStreamer rpt4 DMABuf | 381.5 | 0.24 |
  | 앱 libavcodec DRM_PRIME ([`REPORT.md`](../../../pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.1) | 398.5 | 0.23~0.28 |

  - FPS는 셋 다 HW 디코더 상한(약 380~400)에 있다.
  - CLI의 0.55 ms/f에서 앱보다 많은 몫은 CLI 자체 비용(demux·mux 틀, 동적 링크, `memset`)으로 본다(**추정**).
- **내려받기 방식 사이의 차이는 작다.** NV12(2.14)와 YUV420P(2.21), 미지정(2.02)이 모두 같은 NEON 복사 루프를 쓴다.
- **`vf_unsand`를 쓸 이유는 성능상 없다.** 필터가 하는 일(`av_rpi_sand_to_planar_frame`)은 위 내려받기 경로가 이미 하고 있고, 그 비용이 프레임당 약 1.5 ms다. 프로젝트의 GPU 셰이더(`sand_letterbox`)는 이 복사 없이 SAND dma-buf를 직접 읽는다.

## 주의
- **`vf_unsand`는 이 빌드의 ffmpeg CLI에서 실행할 수 없다**([`evidence/unsand_unusable.txt`](evidence/unsand_unusable.txt)).
  - `-h filter=unsand`에 입력이 `#0 default`, `#1 (null)` 두 개로 나온다. 단순 필터 그래프는 "expected to have exactly 1 input … had 2 input(s)"로 실패한다.
  - 두 번째 입력에 더미를 물린 복합 필터 그래프와, 변환 없이 `hwdownload,format=sand128`만 한 경우도 출력 단계(`wrapped_avframe`)에서 EINVAL로 실패한다.
  - 그래서 "내려받기만"과 "변환"의 비용을 따로 나누지 못했다.
  - rpi 패치의 pad 정의 문제로 보이지만 소스로 확인하지는 않았다(**추정**).
- perf는 조합마다 1회이고 콜 그래프가 없다. `libavutil` 주소를 함수 이름으로 확정하지 못했다.
- 기준선 3회 중 1회(321.42 FPS)가 다른 두 회보다 4% 낮다. 중앙값에는 영향이 없다.
