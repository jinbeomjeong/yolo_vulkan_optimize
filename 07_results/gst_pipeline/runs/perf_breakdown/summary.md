# CPU 차이의 출처: 앱 대 GStreamer 경로, 같은 구간 perf 프로파일

상위 실험 [`gst_pipeline`](../../summary.md)의 회차다. 상위 실험에서 GStreamer 연구 경로가 앱보다 CPU를 약 0.2 ms/frame(6.3%) 덜 썼는데, 그 원인을 찾는다.

## 조건
- 측정 일시: 2026-09-29 08:51~09:40 KST
- 환경: 부트로더 2026-09-25(V3D 1150 MHz), PCIe Gen3, gst-plugins-bad rpt4, perf 6.18.50, git `a26c624`. 자세한 값은 [`env_bench.txt`](env_bench.txt)에 있다.
- 설정: b8, slots 16, 전체 1816프레임. 인자는 상위 실험과 같다(`app_nhcw_b8`, `gst_b8`).
- **구간 일치**
  - perf를 `-D -1 --control fifo:ctl,ack`로 꺼진 채 시작한다. 두 도구가 CPU 측정 구간 경계(c0/c1)에서 `yvo::tool::perf_window()`로 켜고 끈다(`a26c624`).
  - 그 결과 perf의 task-clock이 도구가 낸 CPU ms/f와 **−0.5~−1.0%** 안에서 일치한다(6회).
- **실행**
  - 도구마다 3회를 인터리브하고, 실행마다 `perf stat`(task-clock, cycles, instructions, cs, migrations, page-faults)과 `perf record -e cycles -F 2999`(flat) 두 가지를 돈다.
  - 추가로 캐시 이벤트를 2회씩 잰다. `perf stat`(l1d/l2d refill rd, l3d refill, ll miss rd)과 `perf record -e l3d_cache_refill -c 2000`이다.
  - 모든 실행 전 칩을 65 °C 이하로 식혔다. 20회 모두 exit 0, `hailort.log` warning 0이다.
- 콜 그래프: 앱 300프레임을 `--call-graph dwarf`로 1회 떠서 `memset` 호출자를 찾았다.
- 원본 `perf.data`는 git 밖(`11_logs/2026-09-29_perf_breakdown/`)에 있고, 보고서 텍스트는 [`evidence/`](evidence/)에 있다.

## 결과

### 구간 합계 (`perf stat`, 프레임당)
| 지표 | 앱 (3회) | gst (3회) |
|---|---|---|
| task-clock ms/f (도구 CPU ms/f) | 3.478 / 3.353 / 3.660 (3.509 / 3.382 / 3.688) | 3.024 / 3.119 / 3.132 (3.040 / 3.135 / 3.149) |
| **instructions/f** | **12.64 M** | **12.52 M** (−1%) |
| cycles/f | 6.65 / 6.14 / 6.34 M | 5.86 / 5.72 / 5.75 M |
| IPC | 1.90 / 2.06 / 1.99 | 2.14 / 2.19 / 2.18 |
| context-switches/f | 29.0 / 29.1 / 28.8 | 20.4 / 20.4 / 20.3 |
| page-faults (구간 합) | 3283 | 1903 |

### 어디서 cycles가 다른가 (`perf record` cycles, 3회 중앙값, k cycles/frame)
| DSO | 앱 | gst | 앱 − gst |
|---|---|---|---|
| **libhailort** | **4333** | **3890** | **+443** |
| libc | 573 | 438 | +136 |
| 커널 | 758 | 664 | +93 |
| libavcodec + libavutil + libavformat | 229 | — | |
| GStreamer 계열(libgstbase, libgstreamer, glib, gobject, gstvideo, gstcodecs, codecparsers, coreelements …) | — | ≈ 290 | |
| libvulkan_broadcom | 55 | 46 | +8 |
| **합계** | **6251** | **5614** | **+637** |

| 심볼 | 앱 | gst | 앱 − gst |
|---|---|---|---|
| **libhailort `0x30ffac`** (호스트 NMS 점수 스캔 루프) | **3137** | **2682** | **+456** |
| `__memset_zva64` (libc) | 164 | 11 | +153 |
| `gst_byte_reader_masked_scan_uint32` (h265parse 시작 코드 탐색) | — | 60 | −60 |
| `__memcpy_generic` (libc) | 42 | 97 | −55 |
| libhailort `0x30ffcc` / `0x30ffd4` (같은 루프의 이웃 명령) | 669 / 163 | 674 / 156 | ≈ 0 |

스레드별(comm)
- 앱: 메인과 HailoRT 스레드(이름이 같아 한 줄로 묶임) 5743k, FFmpeg HEVC 프레임 스레드 `av:hevc:df0~4` 543k
- gst: 메인과 HailoRT 스레드 4775k, `qtdemux0:sink`·`q_dec:src`·`q_out:src` 911k

### 캐시 미스 (2회)
| 지표 | 앱 | gst |
|---|---|---|
| L3 refill 합계 (= LL 읽기 미스) | 47.1 / 44.9 M | 47.8 / 47.9 M |
| L2 refill 읽기 | 69.0 / 67.2 M | 70.5 / 71.1 M |
| **NMS 루프 `0x30ffac`의 L3 refill** | **9.67 / 9.56 M** | **9.74 / 9.81 M** |

- `0x30ffac` 부근의 역어셈블은 [`evidence/libhailort_nms_loop_disasm.txt`](evidence/libhailort_nms_loop_disasm.txt)에 있다.
  - `ldr b8, [x2, w0, uxtw]`(0x30ffd0)가 NPU 출력 텐서에서 uint8 점수를 **stride로 건너뛰며 읽는다**.
  - 이어서 `fsub`/`fmul`(역양자화) → `fcmpe`(임계값 비교)를 한다. 루프 카운터는 `w20`이다.
  - 샘플은 루프 머리 `0x30ffac`에 몰린다. 메모리 읽기 stall이 다음 명령으로 넘어가 잡힌 skid로 본다(**추정**).
- `memset` 호출자(dwarf 콜 그래프): **모두 `av:hevc:df*` 스레드 안의 libavcodec 내부 함수**다. 심볼이 제거돼 함수 이름은 알 수 없다.

## 결론
1. **디코드 쪽 코드는 원인이 아니다.** 디코드·역다중화 호스트 비용은 GStreamer 쪽이 오히려 조금 크다(≈290k 대 229k cycles/frame). 상위 summary의 "디코드 쪽 추정"은 틀렸다.
2. **두 경로가 하는 일의 양은 같다.** instructions/frame 차이가 −1%다. 차이는 같은 명령을 실행하는 데 드는 cycles(IPC 1.99 대 2.18)에서 난다.
3. **차이의 약 70%(+456k / +637k cycles/frame)는 HailoRT 호스트 NMS의 점수 스캔 루프 한 곳에서 난다.**
   - 두 경로는 같은 `HailoRunner`를 같은 설정으로 부르고, 이 루프는 같은 횟수로 L3를 놓친다.
   - 그런데 앱에서 cycles가 약 17% 더 든다. 유효 클럭은 비슷하다(약 1.9 GHz).
   - 따라서 **미스 한 번의 메모리 지연이 앱에서 더 길다.** 같은 시간대에 다른 CPU 작업이 메모리를 더 쓰고 있다는 뜻이다(**추정**).
4. 나머지 차이는 대부분 **앱에만 있는 FFmpeg 프레임 스레드의 부담**이다.
   - `memset` +153k cycles/frame(libavcodec 내부, 프레임마다 테이블을 0으로 채우는 것으로 추정)
   - 문맥 교환 +8.6회/frame, 커널 +93k cycles/frame
5. 원인에 대한 해석(**추정**): FFmpeg HEVC 프레임 스레드(`thread_count=0`, 자동으로 코어 수만큼)가 NMS와 같은 시간대에 메모리를 쓰고 깨어나면서 메모리 지연과 스케줄링 비용을 늘린다. GStreamer 경로의 `v4l2slh265dec`에는 이런 프레임 스레드가 없다.
   - 확정하려면 인과 실험이 필요하다. 앱의 FFmpeg 스레드 수를 1로 두고(`FfmpegDrmDecoder::Options::threads = 1`) NMS 루프의 cycles와 CPU ms/f가 gst 수준으로 내려가는지 보면 된다. 명령줄 옵션이 없어 코드 변경이 필요하므로 이 회차에서는 하지 않았다.

## 주의
- perf를 켠 상태의 절대값은 인용하지 않는다. perf record 오버헤드가 있다. 비교는 같은 조건의 두 도구 사이에서만 한다.
- `libhailort`와 `libavcodec` 내부 함수는 심볼이 없어 주소나 호출자 DSO로만 식별했다. `0x30ffac`를 "호스트 NMS 점수 스캔"으로 본 것은 역어셈블 구조에 근거한 판단이다.
- 콜 그래프(dwarf)는 앱 1회 300프레임만 떴다. gst 쪽의 `memcpy`와 h265parse 호출자는 보지 않았다.
- 스레드 이름: HailoRT 내부 스레드는 이름이 없어 프로세스 이름으로 묶인다. 메인 스레드와 HailoRT 스레드를 나누지 못했다.
