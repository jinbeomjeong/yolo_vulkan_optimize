# 확정 구성의 남은 응용 CPU를 perf로 분해 (2026-11-06)

확정 구성 O2의 남은 응용 CPU(1.437 ms/처리 프레임, [`dmabuf_ablation`](../dmabuf_ablation/summary.md)의 `base`)가 어디에 쓰이는지 나눈다.
부모 실험은 [`technique_ablation`](../../summary.md)이다. 유형은 프로브형이다(근거는 `evidence/`).

## 측정 조건

- 날짜: 2026-11-06 18:46(KST). 환경은 [`env_bench.txt`](env_bench.txt)에 있다.
- Raspberry Pi CM5, Hailo-8 PCIe Gen3 x1, V3D 1150 MHz, HailoRT 4.24.0, YOLO11m, `video_h265.mp4`(1080p30, 1816프레임).
- **CPU 거버너 performance**로 측정했다. 측정 동안만 바꿨고, 끝난 뒤 ondemand로 돌아온 것을 확인했다.
- 바이너리 md5는 `ef24741c7e5ca9457253f73e03aa73ca`로, `dmabuf_ablation`과 같은 바이너리다.
- 구성은 `technique_ablation_bench.sh`의 `base`와 같다.
  - 배치 1/슬롯 1, 입력 페이싱 source, 입력 큐 1.
  - 표시 on(Vulkan 합성, wayland 전체 화면), 기록 on.
  - NHCW dma-buf 입력, GPU 후처리, GPU 억제(NMS).
  - 디코더는 HW, 1스레드.
- perf 설정: `perf record -g --call-graph=fp -F 999 -D -1 --control fifo:ctl,ack`. 도구가 `YVO_PERF_CTL`/`YVO_PERF_ACK`로 알리는 CPU 창에서만 기록했다.
- 3회 실행했고, Hailo 칩 시작 온도는 55.4·64.1·64.9 °C였다. 측정 스크립트는 저장소에 넣지 않았으며, 위 명령에 냉각 대기만 더한 것이다.

## 결과

perf를 켜면 응용 CPU가 1.678·1.562·1.579 ms/처리 프레임이 된다. 끈 기준값(1.437)보다 약 0.14 ms 높다.
처리 FPS는 24.28~24.30이고, hailort.log의 warning·error는 0건이다.
아래 ms는 경로 비중에 그 실행의 ms/처리 프레임을 곱한 값이다. 실행당 표본은 약 1000개라 1% 미만 항목은 상대 오차가 ±30~50%다.
원표는 [`evidence/vulkan_path_table.txt`](evidence/vulkan_path_table.txt)에 있다.

| 경로 | 중앙값 ms/처리 프레임 | 비중 |
|---|---:|---:|
| 메인 스레드 전체 | 1.008 | 64% |
| └ libavcodec(HEVC 디코드 호출) | 0.242 | 15% |
| 　└ `__memset_zva64`(FFmpeg `hevc_frame_start`의 표 초기화) | 0.088 | 5.7% |
| └ 결과 처리 람다(`flush_results`, 표시·jsonl 포함) | 0.246 | 16% |
| 　└ `DisplayPipeline::show`(present 0.128 포함) | 0.220 | 14% |
| └ `DisplayPipeline::pump`(present 완료 폴링) | 0.100 | 6% |
| └ `YoloPostproc::record`(후처리 명령 버퍼 재기록) | 0.083 | 5% |
| └ `SandLetterbox::record`(전처리 명령 버퍼 재기록) | 0.057 | 3.6% |
| V3DV 드라이버 제출 스레드 | 0.269 | 17% |
| └ `v3d_submit_cpu_ioctl`(타임스탬프·쿼리 리셋이 만드는 CPU 잡) | 0.086 | 5.5% |
| HailoRT 스레드 2개 | 0.276 | 17% |
| └ dma-buf map·unmap ioctl(래퍼 포함) | 0.042 | 2.7% |
| V4L2 디코드 보조 스레드 | 0.028 | 2% |

HailoRT 쪽 커널 경로는 [`evidence/hailo_resolved_run{1,2,3}.txt`](evidence/)에 있다.

## 결론

- **남은 CPU의 큰 몫은 연산이 아니라 드라이버와 런타임의 잡 단위 비용이다.**
  - V3DV 제출 스레드, 명령 버퍼 재기록(BO의 mmap·munmap), HailoRT 런타임, 표시의 present·폴링이 여기에 든다.
  - 한 덩어리로 0.3 ms를 넘는 것은 없다.
- **HailoRT가 프레임마다 dma-buf를 다시 매핑한다.**
  - 원인은 우리 코드의 크기 계약이다. `HailoRawRunner::bind_slot`은 사전 매핑을 dma-buf 전체 크기로 하고, `infer_async`에는 텐서 크기(hw_frame_size)를 넘긴다.
  - HailoRT 4.24의 매핑 캐시는 `{fd, 방향, 크기}`가 정확히 같아야 적중하므로 적중하지 못한다(hailo-expert 소스 분석).
  - 비용은 약 0.04 ms/처리 프레임이다. 프레임당 map·unmap 횟수(입력 1 + 출력 6으로 추정)는 세지 않았다.
- **후처리 명령 버퍼를 프레임마다 다시 기록한다.**
  - 입력이 슬롯별로 바뀌지 않아 사전 기록이 가능하다.
  - 재기록 비용의 대부분은 V3DV가 BO를 해제할 때 munmap하고 다시 mmap하는 데서 나온다(Mesa 26.2.2 `v3dv_bo.c`).
- **GPU 타임스탬프가 CPU 잡을 만든다.**
  - V3DV에서 `vkCmdWriteTimestamp2`와 `vkCmdResetQueryPool`은 CPU 잡이 된다. 프레임당 약 19개로 추정하며, 소스를 읽고 센 값이다.
  - 제출 스레드 비용의 가장 큰 항목이다.
- **표시 경로에도 줄일 후보가 있다.**
  - present 안에서 sync file 생성을 기다리는 `sched_yield` 스핀이 있다.
  - 표시 on일 때 500 µs 단위로 깨어나 present 완료를 폴링한다.
  - 논문 §7.4 (4)의 "이 경로에서 줄일 수 있는 몫은 찾지 못하였다"는 이 실험 전 판단이다.
- **HEVC 디코드 호출의 memset 0.088 ms는 시스템 libavcodec 안이다.** 우리 코드로는 없앨 수 없다.

## 주의·남은 일

- **perf 커널 모듈 심볼 오해석:** 이 보드에서 `perf report`는 hailo_pci·drm·v3d·gpu_sched 모듈 심볼을 잘못 푼다(root로 풀어도 같다).
  - 예: `drm_mode_create_lease_ioctl`로 표시된 것은 실제로 `drm_ioctl`이고, `board_location_show`는 `hailo_pcie_fops_unlockedioctl`이다.
  - 위 표는 `perf script`의 커널 주소를 같은 부팅의 `/proc/kallsyms`로 다시 풀어 만들었다. 커널 본체 심볼은 정상이다.
- libvulkan_broadcom과 libhailort는 심볼이 없다(stripped).
  - 드라이버 함수 연결은 Mesa 26.2.2 소스와 커널 호출 대상에서 추론한 것이다.
  - HailoRT 사용자 공간 약 0.28 ms의 내부는 나누지 못했다.
- 절감 후보와 추정치(perf 켠 척도, ms/처리 프레임)는 다음과 같다. 모두 측정 전 추정이다.
  - 타임스탬프 끄기 0.10~0.13
  - 후처리 사전 기록 0.08
  - HailoRT 사전 매핑 크기 맞춤 0.04
  - present 폴링 줄이기 0.04~0.06
  - present 호출 늦추기 0.03~0.05
  - 전처리 명령 버퍼 캐시 약 0.05
- `perf.data`는 `.gitignore` 대상이라 저장소에 없다(보드에만 있다).
