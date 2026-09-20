# 영상 디코드 벤치마크: H.264 SW vs H.265 HW/SW

- 측정 일시: 2026-09-20 01:23~02:16 KST(1차), 02:21~02:26 KST(나머지 조합 재개)
- 환경: Raspberry Pi CM5 (BCM2712, Cortex-A76 4코어 1.5 GHz, governor ondemand), GStreamer 1.26.2, kernel 6.18.50
- 스크립트: [`06_tools/bench/decode_bench.sh`](../../06_tools/bench/decode_bench.sh). 조합마다 3회를 실행하고 중앙값을 쓴다. CPU 시간은 wait4 rusage에서 얻는다(`/usr/bin/time -v`와 같은 값)
- 입력: `04_data/videos/video_h264.mp4`(H.264 High 1080p30, 1834프레임), `video_h265.mp4`(HEVC Main 1080p30, 1816프레임), `urban_street.mp4`(H.264 Main 720p30, 앞 3000프레임)
- 단계
  - S1: 디코드만 한다 (`fakesink sync=false`)
  - S2: 디코드한 뒤 `videoconvertscale add-borders=true`로 letterbox를 적용하고 `RGB 640x640`으로 변환한다. NPU 입력 형태와 같다
- 모든 실행에서 전체 프레임을 디코드했다. 같은 조합 3회의 FPS 편차는 최대 2.6%다
- 1차 측정에서 `h265_D_t0_S2_nt1`는 1회만 끝났다. 그래서 이 조합은 재개 때 잰 3회 값만 쓴다

## 결과 (1080p, 중앙값)

| 디코더 | S1 FPS | S1 CPU (4코어 기준) | S2 FPS (변환 n-threads=4) | 프레임당 CPU 시간 S1 / S2 |
|---|---:|---:|---:|---:|
| avdec_h264 자동 스레드 | 164.6 | 83% | 116.4 | 20.2 / 28.1 ms |
| avdec_h264 1스레드 | 59.0 | 25% | 51.6 | 17.0 / 25.2 ms |
| avdec_h264 2스레드 | 107.7 | 49% | 86.7 | 18.2 / 26.4 ms |
| openh264dec | 31.4 | 25% | 29.1 | 31.9 / 40.2 ms |
| **v4l2slh265dec (HW, GStreamer)** | **32.7** | **25% (1코어 포화)** | 30.3 | 30.6 / 39.4 ms |
| avdec_h265 자동 스레드 | 109.1 | 87% | 84.2 | 31.8 / 40.0 ms |
| ffmpeg HW HEVC → 파이프 → GStreamer | 146.9 | 43% (2개 프로세스 합) | 109.2 | 11.7 / 23.6 ms |
| 참고: ffmpeg `-hwaccel drm` HEVC 단독 | 332.8 | 17% | - | 2.0 ms |

720p H.264 (`urban_street`, avdec_h264 자동 스레드)
- S1: 397.8 FPS, 7.4 ms/frame
- S2: 209.7 FPS (n-threads=4), 13.7 ms/frame

- 원시 데이터: `results_raw.csv`(실행별), `results.csv`(조합별 중앙값), `raw/`(gst 로그)
- `results_raw_part1.csv`는 1차 측정 원본이다
- 스레드 0(auto)과 4의 결과는 같다. auto는 4코어를 모두 쓴다

## 해석

**1. HW HEVC 디코더 자체는 빠르지만, GStreamer 경로에서는 CPU 복사가 병목이다**
- ffmpeg(`-hwaccel drm`, 출력은 null muxer)로 디코드하면 333 FPS, CPU 2 ms/frame이 나온다. 하드웨어는 1080p를 약 11배속으로 디코드한다
- 같은 하드웨어를 GStreamer `v4l2slh265dec`로 쓰면 32.7 FPS다. 이때 CPU 코어 하나가 100% 찬다
- perf 프로파일로는 CPU 시간의 약 90%가 ORC가 생성한 프레임 복사·변환 코드였다. 디코더 출력(NV12_128C8, SAND 타일 포맷)을 CPU 한 스레드로 시스템 메모리에 옮기는 비용으로 추정한다
- VideoMeta를 지원하는 `fakevideosink`로 바꿔도 결과는 같았다(32.6 FPS)
- 현재 GStreamer 경로에서는 HW 디코드를 써도 CPU 비용(30.6 ms/frame)이 SW H.265(31.8 ms/frame)와 거의 같다

**2. H.264 SW 디코드는 30 FPS와 52 FPS를 모두 감당한다. 다만 CPU를 많이 쓴다**
- avdec_h264로 1080p를 디코드하고 변환까지 하면 약 28 ms/frame이다
  - 30 FPS에서 약 0.84코어(4코어의 21%)를 쓴다
  - NPU 상한인 52 FPS에서는 약 1.45코어(36%)를 쓴다
- openh264dec는 단일 스레드라 약 31 FPS에서 멈춘다. 1080p 30 FPS를 버틸 여유가 없다

**3. 640x640 RGB 변환만으로 1080p 기준 약 7~8 ms/frame의 CPU가 든다**
- 변환 스레드를 1에서 4로 늘리면 처리량은 오르지만 CPU 총량은 같다
- 이 단계는 Vulkan 전처리로 옮길 1순위 후보다

**4. 720p 입력은 1080p보다 디코드가 약 2.7배 싸다**
- 720p는 7.4 ms/frame, 1080p는 20.2 ms/frame이다
- 입력 해상도를 줄이는 것만으로 CPU 부담이 크게 줄어든다

## 다음 단계 제안
- ffmpeg HW 디코드를 GStreamer로 넘기는 방법 조사: [notes/ffmpeg_to_gstreamer.md](notes/ffmpeg_to_gstreamer.md). 파이프 방식은 3회 측정 결과 디코드만 11.7 ms/frame, 변환 포함 23.6 ms/frame으로 GStreamer HW 경로(30.6 / 39.4 ms)와 avdec_h264(20.2 / 28.1 ms)보다 싸다. ffmpeg에 `-fps_mode passthrough`가 필요하다
- **HW HEVC zero-copy 경로 확인:** `v4l2slh265dec`는 DMABuf 출력에서도 내부 복사가 일어난다(조사 문서 참고). 그래서 libavcodec의 DRM_PRIME 프레임(dma-buf fd)을 Vulkan으로 import하고, SAND→RGB letterbox를 GPU에서 처리한다. 성공하면 디코드와 전처리의 CPU 비용이 약 2 ms/frame 수준까지 줄 수 있다. Hailo↔Vulkan 버퍼 공유와 같은 인터페이스 설계가 필요하다
- **GStreamer 복사 원인 확정:** DMABuf caps를 강제해도 복사가 남는 것은 확인했다(`notes/ffmpeg_to_gstreamer.md`). 남은 일은 v4l2codecs 소스에서 copy 판단 조건을 찾는 것이다

## 측정 조건 메모
- 매 실행 전 SoC 온도가 기준 이하로 내려갈 때까지 기다렸다. 1차는 65 °C·최대 120초, 재개는 70 °C·최대 30초였다
- 측정 직후 온도는 67~83 °C였다
- `get_throttled`는 0x80000이었다. 과거에 soft temp limit이 발생했다는 기록 비트이고, 측정 시점에는 스로틀링이 없었다
- 측정 중 NPU·GPU 작업은 없었다
