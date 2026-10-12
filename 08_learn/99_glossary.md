# 99. 용어집과 다음에 읽을 것

> 이 챕터는 처음부터 읽는 문서가 아니다. **다른 챕터를 읽다 막혔을 때 찾아보는 사전**이다.
> 각 항목 끝의 `→ 0X` 는 그 용어를 제대로 다루는 챕터다.

---

## A. 성능·측정 용어

| 용어 | 뜻 |
|---|---|
| **처리량 (throughput)** | 1초에 몇 장을 처리하느냐. 단위 FPS → [01](01_big_picture.md) |
| **FPS** | frames per second, 초당 프레임 수 → [01](01_big_picture.md) |
| **지연 (latency)** | 한 장이 들어가서 나올 때까지 걸리는 시간 → [01](01_big_picture.md) |
| **e2e 지연** | end-to-end. 프로그램 안에서 프레임이 들어와 결과가 나올 때까지 → [01](01_big_picture.md) |
| **glass-to-glass** | 카메라 유리에 비친 장면이 모니터 유리에 나타날 때까지. 사람이 체감하는 진짜 지연 → [01](01_big_picture.md) |
| **병목 (bottleneck)** | 파이프라인에서 가장 느린 단계. 전체 처리량을 정한다 → [01](01_big_picture.md) |
| **파이프라이닝** | 여러 장을 동시에 서로 다른 단계에서 처리하는 것. 처리량은 늘고 지연은 는다 → [01](01_big_picture.md) |
| **in-flight** | 지금 파이프라인 안에 들어가 있는 프레임 수. `≈ FPS × 지연`(Little's law) → [04](04_npu_hailort.md) |
| **운용점 (operating point)** | batch·slots 등을 어떻게 정할지의 선택. 처리량과 지연을 맞바꾼다. 실시간 표시 `b2/s4`, 최대 처리량 `b8/s16`(단시간), 최소 지연 `b1/s1` + 입력 페이싱 큐 1 → [01](01_big_picture.md), [10](10_why_this_design.md) |
| **입력 페이싱 (input pacing)** | 파일 입력을 카메라처럼 소스 시각(30 fps 면 33.3 ms 간격)에 맞춰 넣는 것. `--input-pace source`, 표시 off 전용 → [04](04_npu_hailort.md), [08](08_method.md) |
| **`e2e_arr`** | 프레임이 **도착한 시각**부터 결과까지의 지연. 입력 페이싱에서만 잰다. 디코드 끝부터 재는 e2e 와 섞지 않는다 → [08](08_method.md) |
| **leaky 큐** | 넘치면 가장 오래된 항목을 버리는 크기 제한 큐(GStreamer `queue leaky=downstream`). `--input-queue N`, 이 프로젝트는 1이 맞다 → [04](04_npu_hailort.md) |
| **중앙값 (median)** | 값을 정렬했을 때 가운데 값. 이 프로젝트는 평균 대신 이걸 쓴다 → [08](08_method.md) |
| **p90** | 90번째 백분위수. 100번 중 90번은 이 값보다 좋다는 뜻 → [08](08_method.md) |
| **클럭 강등 (throttling)** | 칩이 뜨거워져 동작 속도를 스스로 낮추는 것 → [02](02_hardware.md), [08](08_method.md) |
| **대조군 (control)** | 바꾸지 않은 기준 설정. 오늘 환경이 과거와 같은지 확인하는 용도 → [08](08_method.md) |
| **perf** | 어느 함수가 CPU 시간을 쓰는지 보여 주는 리눅스 도구 → [08](08_method.md) |
| **ASan / UBSan** | 메모리 오류·정의되지 않은 동작을 잡는 컴파일러 기능 → [08](08_method.md) |
| **`perf_window`** | perf를 도구가 보고하는 CPU 구간에서만 켜는 헬퍼(`YVO_PERF_CTL`/`YVO_PERF_ACK` FIFO, `perf -D -1 --control`) → [08](08_method.md), [09](09_code_map.md) |

---

## B. 하드웨어

| 용어 | 뜻 |
|---|---|
| **SoC** | System on Chip. CPU·GPU·디코더 등 여러 블록이 한 칩에 든 것. 이 보드는 BCM2712 → [02](02_hardware.md) |
| **CPU** | 이 보드는 Cortex-A76 코어 4개 → [02](02_hardware.md) |
| **GPU / V3D** | VideoCore VII. Vulkan 드라이버 이름이 V3DV → [02](02_hardware.md), [05](05_vulkan_compute.md) |
| **NPU** | Neural Processing Unit. 신경망 전용 칩. 이 보드는 Hailo-8 → [04](04_npu_hailort.md) |
| **ISP** | Image Signal Processor. 이 보드는 PiSP Back End. 구현됐으나 **미사용** → [02](02_hardware.md), [10](10_why_this_design.md) |
| **PCIe** | 칩과 칩을 잇는 고속 통로. 세대(Gen2/Gen3)와 레인 수(x1)가 대역폭을 정한다 → [02](02_hardware.md) |
| **GT/s** | giga-transfers per second. PCIe 링크 속도 단위. Gen2 = 5.0, Gen3 = 8.0 → [02](02_hardware.md) |
| **타일 기반 렌더러** | 화면을 작은 타일로 쪼개 칩 안 메모리에서 그리는 GPU 구조. 렌더패스를 열면 프레임버퍼가 왕복한다 → [02](02_hardware.md), [10](10_why_this_design.md) |
| **통합 메모리** | CPU·GPU·NPU가 같은 DRAM을 쓰는 구조. zero-copy의 전제 → [02](02_hardware.md) |
| **SIMD** | Single Instruction Multiple Data. 명령 하나로 여러 데이터를 동시에 처리하는 방식. GPU가 대량 반복에 강한 이유 → [02](02_hardware.md) |
| **거버너 (governor)** | CPU 클럭을 부하에 따라 올리고 내리는 커널 정책. 측정값을 흔드는 변수 중 하나 → [02](02_hardware.md) |
| **부트로더 EEPROM / `v3d_freq`** | 보드 펌웨어와 그것이 정하는 V3D(GPU) 클럭. 2026-09-25 부트로더가 기본값을 960 → 1150 MHz로 올려 GPU ms가 바뀌었다. `vcgencmd get_config v3d_freq`로 읽는다 → [05](05_vulkan_compute.md), [08](08_method.md) |

---

## C. 메모리와 버퍼 공유

| 용어 | 뜻 |
|---|---|
| **DMA** | Direct Memory Access. 장치가 CPU를 거치지 않고 메모리를 직접 읽고 쓰는 것 → [06](06_zerocopy_dmabuf.md) |
| **캐시 (cache)** | 자주 쓰는 메모리 값을 CPU 가까이 두는 것. 장치와 공유하면 일관성 문제가 생긴다 → [02](02_hardware.md), [06](06_zerocopy_dmabuf.md) |
| **일관성 (coherency)** | 캐시에 든 값과 메모리의 값이 어긋나지 않는 성질 → [06](06_zerocopy_dmabuf.md) |
| **write-combine** | 쓰기를 모아서 한 번에 내보내는 매핑. 쓰기는 빠르고 읽기는 느리다 → [06](06_zerocopy_dmabuf.md) |
| **`HOST_CACHED` / `HOST_COHERENT`** | Vulkan 메모리 타입 플래그. **V3D에는 `HOST_CACHED`가 없다** — 이 프로젝트 설계를 지배하는 사실 → [06](06_zerocopy_dmabuf.md), [10](10_why_this_design.md) |
| **zero-copy** | 단계 사이에 데이터를 복사하지 않고 같은 메모리를 여러 장치가 직접 보는 것 → [01](01_big_picture.md), [06](06_zerocopy_dmabuf.md) |
| **dma-buf** | 커널이 주는 버퍼 공유 fd. zero-copy의 수단 → [06](06_zerocopy_dmabuf.md) |
| **fd (파일 디스크립터)** | 커널 자원을 가리키는 정수 손잡이 → [06](06_zerocopy_dmabuf.md) |
| **DRM_PRIME** | 디코더·GPU가 버퍼를 dma-buf fd로 주고받는 방식 → [06](06_zerocopy_dmabuf.md), [07](07_video_display.md) |
| **dma-heap** | 유저 공간에서 dma-buf를 직접 할당하는 커널 인터페이스(`/dev/dma_heap/...`) → [06](06_zerocopy_dmabuf.md) |
| **external memory (Vulkan)** | 바깥에서 만든 메모리를 Vulkan으로 들여오거나 내보내는 확장 → [06](06_zerocopy_dmabuf.md) |
| **import / export** | 남이 만든 fd를 Vulkan 버퍼로 들여오는 것 / Vulkan 메모리를 fd로 내보내는 것 → [06](06_zerocopy_dmabuf.md) |
| **정렬 (alignment)** | 주소나 크기가 특정 배수여야 하는 제약. HailoRT는 16384 B를 요구한다 → [06](06_zerocopy_dmabuf.md) |
| **`DMA_BUF_IOCTL_SYNC`** | CPU가 dma-buf를 만지기 전후에 커널에 알리는 호출(begin/end). 캐시 동기화의 계약 → [06](06_zerocopy_dmabuf.md) |
| **ownership transfer** | 버퍼를 누가 쓰는지가 바뀔 때(바깥 장치 ↔ GPU 큐) 넣어야 하는 Vulkan 배리어 → [06](06_zerocopy_dmabuf.md) |
| **stride** | 다음 행(또는 다음 요소)까지의 바이트 간격 → [03](03_yolo_tensor.md), [07](07_video_display.md) |

---

## D. GPU / Vulkan

| 용어 | 뜻 |
|---|---|
| **Vulkan** | 저수준 그래픽스·컴퓨트 API → [05](05_vulkan_compute.md) |
| **컴퓨트 셰이더 (compute shader)** | 그림 그리기가 아니라 **계산**을 시키는 GPU 프로그램. 확장자 `.comp` → [05](05_vulkan_compute.md) |
| **GLSL** | 셰이더를 쓰는 언어 → [05](05_vulkan_compute.md) |
| **SPIR-V** | GLSL을 컴파일한 중간 표현. GPU가 실제로 받는 것 → [05](05_vulkan_compute.md) |
| **워크그룹 (workgroup)** | 셰이더 실행 단위 묶음. 크기를 어떻게 잡느냐가 성능을 크게 바꾼다 → [05](05_vulkan_compute.md) |
| **invocation** | 셰이더의 한 번 실행. 보통 픽셀 하나나 요소 몇 개를 맡는다 → [05](05_vulkan_compute.md) |
| **subgroup** | 하드웨어가 한꺼번에 돌리는 invocation 묶음. V3D는 16 → [05](05_vulkan_compute.md) |
| **레지스터 (register)** | 계산 장치 안의 아주 작은 저장소. 명령이 바로 읽어 대기가 없다. V3D 7.1은 스레드당 rf0~rf63(2스레드) 또는 32개(4스레드) → [05](05_vulkan_compute.md) §5.9 |
| **spill / fill** | 레지스터가 모자라 컴파일러가 값을 메모리에 내려 두고(spill) 다시 읽는 것(fill). 메모리 왕복이 늘어 느려진다. `V3D_DEBUG=shaderdb`로 개수를 본다 → [05](05_vulkan_compute.md) §5.9 |
| **SWAR** | SIMD Within A Register. 32비트 하나를 바이트 4칸으로 보고 한 번의 정수 연산으로 4개를 동시에 비교·누적하는 기법(`yolo_decode`의 `ge_cnt`) → [05](05_vulkan_compute.md) §5.9 |
| **argmax 지름길** | `yolo_decode` ⑤에서 통과 클래스가 1개면 ①이 레지스터에 남긴 `pmax4`로 바로 쓰고 클래스 텐서를 다시 읽지 않는 것 → [05](05_vulkan_compute.md) §5.9 |
| **디스패치 (dispatch)** | 컴퓨트 셰이더를 실제로 돌리라고 시키는 명령 → [05](05_vulkan_compute.md) |
| **SSBO** | Shader Storage Buffer Object. 셰이더가 읽고 쓰는 큰 버퍼 → [05](05_vulkan_compute.md) |
| **descriptor set** | 셰이더가 볼 버퍼·이미지들을 미리 묶어 둔 것 → [05](05_vulkan_compute.md) |
| **push constant** | 셰이더에 넘기는 작은 값들(수십~백여 바이트). 매 디스패치마다 바꿀 수 있다 → [05](05_vulkan_compute.md) |
| **specialization constant** | 셰이더 컴파일 시점에 호스트가 정하는 상수. 워크그룹 크기·모드 스위치에 쓴다 → [05](05_vulkan_compute.md) |
| **배리어 (barrier)** | "여기까지 끝난 다음에 다음을 해라"를 GPU에 알리는 것 → [05](05_vulkan_compute.md) |
| **fence / semaphore** | GPU 작업 완료를 기다리는 동기화 객체. fence는 CPU가, semaphore는 GPU가 기다린다 → [05](05_vulkan_compute.md) |
| **timestamp query** | GPU가 각 구간에 쓴 시간을 재는 수단 → [05](05_vulkan_compute.md), [08](08_method.md) |
| **성능 카운터 (`VK_KHR_performance_query`)** | GPU 하드웨어 카운터(QPU-active, 메모리 읽기 대기 등)를 읽는 확장. 시간이 **왜** 걸리는지 본다. 한 쿼리가 안의 잡을 합산하고 다른 클라이언트 잡이 섞인다 → [05](05_vulkan_compute.md) |
| **Vulkan Video** | 전용 비디오 엔진으로 디코드·인코드하는 Vulkan 확장(`VK_KHR_video_*`). V3DV에는 **없다** → [05](05_vulkan_compute.md), [07](07_video_display.md) |
| **`shaderInt64`** | 셰이더에서 64비트 정수를 쓸 수 있는지. V3D는 false라 자체 셰이더는 int32로만 짠다 → [05](05_vulkan_compute.md), [10](10_why_this_design.md) |
| **검증 레이어 (validation layer)** | Vulkan 사용법이 틀렸을 때 알려 주는 디버그 계층 → [05](05_vulkan_compute.md) |
| **VUID** | Valid Usage ID. 검증 레이어가 인용하는 스펙 규칙 번호 → [05](05_vulkan_compute.md) |
| **storage image** | 셰이더가 직접 픽셀을 써넣을 수 있는 이미지 → [07](07_video_display.md) |
| **dispatch indirect** | 워크그룹 개수를 CPU가 아니라 **버퍼에 적힌 값**에서 읽어 오는 디스패치 → [05](05_vulkan_compute.md) |

---

## E. 영상과 표시

| 용어 | 뜻 |
|---|---|
| **컨테이너 / 코덱** | mp4는 담는 그릇(컨테이너), H.264·HEVC는 압축 방식(코덱) → [07](07_video_display.md) |
| **HEVC (H.265)** | 압축 코덱. 이 보드는 **이것만 하드웨어로 푼다** → [07](07_video_display.md) |
| **디코드 (decode)** | 압축된 영상을 그림으로 푸는 것 → [07](07_video_display.md) |
| **프레임 스레딩 (frame threading)** | 디코더가 여러 프레임을 스레드 여러 개로 동시에 푸는 방식. 처리량은 늘지만 CPU·지연이 는다. 앱은 2026-09-29부터 1스레드(`YVO_DEC_THREADS`) → [07](07_video_display.md), [10](10_why_this_design.md) |
| **`v4l2slh265dec`** | GStreamer의 HW HEVC 디코더 요소(v4l2codecs, stateless). rpt3 패키지에서는 1080p에서 CPU 복사로 빠졌고 rpt4에서 고쳐졌다 → [07](07_video_display.md) §7.7 |
| **YUV / RGB** | 색을 표현하는 두 방식. 영상은 YUV, 화면은 RGB → [07](07_video_display.md) |
| **4:2:0 크로마 서브샘플링** | 색 정보를 밝기보다 성기게 저장하는 것. 사람 눈이 색 해상도에 둔한 성질을 이용 → [07](07_video_display.md) |
| **NV12** | Y 평면 + UV 인터리브 평면으로 된 4:2:0 배치 → [07](07_video_display.md) |
| **SAND128** | Broadcom 전용 타일 배치. 폭 128바이트 열로 쪼개 놓았다 → [07](07_video_display.md) |
| **BT.709 limited range** | HD 영상의 색 변환 규격과 값 범위(0~255가 아니라 16~235) → [07](07_video_display.md) |
| **sRGB / 감마** | 화면에 내보낼 때 쓰는 비선형 밝기 인코딩 → [07](07_video_display.md) |
| **LSB** | Least Significant Bit. 8비트 픽셀 값의 최소 단위 1. 화질 오차를 "몇 LSB"로 센다 → [07](07_video_display.md) |
| **letterbox** | 비율을 유지한 채 정사각형 안에 넣고 남는 곳을 채우는 것(이 프로젝트는 값 114) → [07](07_video_display.md) |
| **Wayland** | 리눅스의 창 관리 시스템. 컴포지터가 창들을 합성한다 → [07](07_video_display.md) |
| **스왑체인 (swapchain)** | 화면에 내보낼 이미지들의 회전 목록 → [07](07_video_display.md) |
| **present** | 그린 이미지를 화면에 내보내는 것 → [07](07_video_display.md) |
| **present mode (FIFO / MAILBOX / IMMEDIATE)** | 화면 갱신과 어떻게 맞출지. FIFO는 vsync에 맞춰 대기 → [07](07_video_display.md) |
| **vsync / tearing** | 화면 갱신 주기에 맞추는 것 / 안 맞췄을 때 화면이 찢어져 보이는 현상 → [07](07_video_display.md) |
| **HVS** | Pi 5 의 디스플레이 컨트롤러(Hardware Video Scaler). 평면 여러 장을 스캔아웃하면서 스케일·YUV→RGB·알파 블렌딩을 한다. GPU(V3D)와 별개 → [07](07_video_display.md) §7.12 |
| **평면 (plane) / 스캔아웃 (scanout)** | 디스플레이 컨트롤러가 겹쳐 내보내는 버퍼 한 장 / 버퍼를 읽어 모니터 신호로 내보내는 것 → [07](07_video_display.md) §7.12 |
| **direct scanout (zero-copy 스캔아웃)** | 컴포지터가 합성하지 않고 클라이언트 버퍼를 그대로 평면에 올리는 것 → [07](07_video_display.md) §7.12 |
| **KMS / atomic commit** | 리눅스 커널의 화면 설정 API(Kernel Mode Setting) / 평면·CRTC 속성 여러 개를 한 번에 적용하는 요청 → [07](07_video_display.md) §7.12 |
| **CRTC / 커넥터** | KMS 에서 스캔아웃 타이밍을 만드는 단위 / HDMI 같은 출력 단자 → [07](07_video_display.md) §7.12 |
| **DRM master** | 화면 설정(모드셋·평면 커밋)을 할 권한을 가진 단 하나의 프로세스. 평소 컴포지터(labwc)가 쥔다 → [07](07_video_display.md) §7.12 |
| **VT (가상 터미널)** | Ctrl+Alt+F1~ 로 바꾸는 콘솔/세션 화면. 데스크톱이 비활성 VT 로 가면 master 를 놓는다(`loginctl activate`, `chvt`) → [07](07_video_display.md) §7.12 |
| **subsurface** | Wayland 에서 부모 surface 위에 붙는 자식 surface. 박스 평면을 이것으로 올린다 → [07](07_video_display.md) §7.12 |
| **zpos** | 평면의 겹침 순서. 클수록 위 → [07](07_video_display.md) §7.12 |
| **AB24 (ABGR8888)** | DRM 픽셀 포맷 이름. 메모리 바이트 순서가 R,G,B,A 라 Vulkan `R8G8B8A8` 과 같다 → [07](07_video_display.md) §7.12 |
| **DRM modifier** | 버퍼의 타일 배치를 나타내는 64비트 값(LINEAR, SAND128 등) → [06](06_zerocopy_dmabuf.md), [07](07_video_display.md) |
| **COLOR_ENCODING / COLOR_RANGE** | YUV 평면의 변환 계수(BT.601/709)와 값 범위(limited/full)를 정하는 평면 속성. Wayland 에서는 color-representation 프로토콜로 요청한다 → [07](07_video_display.md) §7.12 |
| **debugfs (`/sys/kernel/debug/dri/1/state`)** | 커널이 보여 주는 현재 KMS 상태. 평면에 무엇이 올라갔는지를 판정하는 근거 → [07](07_video_display.md) §7.12 |

---

## F. 신경망과 YOLO

| 용어 | 뜻 |
|---|---|
| **텐서 (tensor)** | 숫자를 여러 차원으로 늘어놓은 배열 → [03](03_yolo_tensor.md) |
| **레이아웃 (layout)** | 텐서의 차원을 메모리에 어떤 순서로 늘어놓는지 → [03](03_yolo_tensor.md) |
| **NCHW / NHWC / NHCW** | 세 가지 레이아웃. N=배치, C=채널, H=높이, W=너비. **이 프로젝트의 핵심 주제** → [03](03_yolo_tensor.md) |
| **CNN** | Convolutional Neural Network. 이미지를 다루는 대표적 신경망 구조 → [03](03_yolo_tensor.md) |
| **YOLO** | You Only Look Once. 한 번에 객체 위치와 종류를 내놓는 검출 모델 계열 → [03](03_yolo_tensor.md) |
| **추론 (inference)** | 학습이 끝난 모델을 실제로 돌려 결과를 얻는 것 → [04](04_npu_hailort.md) |
| **anchor-free 그리드** | 격자칸마다 직접 박스를 예측하는 방식 → [03](03_yolo_tensor.md) |
| **P3 / P4 / P5** | 크기가 다른 격자 세 개(80×80 / 40×40 / 20×20). 작은 물체부터 큰 물체까지 담당 → [03](03_yolo_tensor.md) |
| **stride** | 격자칸 하나가 원본에서 몇 픽셀인지(8 / 16 / 32) → [03](03_yolo_tensor.md) |
| **DFL** | Distribution Focal Loss. 거리를 값 하나가 아니라 **분포**로 예측하고 기댓값을 취한다 → [03](03_yolo_tensor.md) |
| **reg_max** | DFL 분포의 구간 수. 이 모델은 16 → [03](03_yolo_tensor.md) |
| **softmax** | 숫자들을 합이 1인 확률로 바꾸는 함수 → [03](03_yolo_tensor.md) |
| **sigmoid** | 숫자를 0~1로 누르는 함수. 이 모델은 칩이 클래스 출력에 **이미 적용**했으므로 **호스트는 다시 적용하지 않는다** → [03](03_yolo_tensor.md), [12](12_hailo8_operator_support.md) |
| **IoU** | Intersection over Union. 두 박스가 얼마나 겹치는지(0~1) → [03](03_yolo_tensor.md) |
| **NMS** | Non-Maximum Suppression. 겹친 박스 중 점수 높은 것만 남기는 것 → [03](03_yolo_tensor.md) |
| **EfficientNMS** | TensorRT의 GPU NMS 플러그인. Filter → Sort → 타일+상태 방송 구조. 이 프로젝트 `efficient_nms.comp`의 설계 원형 → [11](11_tensorrt_efficient_nms.md), [03](03_yolo_tensor.md) |
| **양자화 (quantization)** | 실수를 정수(여기서는 8비트)로 줄여 저장하는 것. `값 = (q − zero_point) × scale` → [03](03_yolo_tensor.md) |
| **zero-point / scale** | 양자화를 되돌리는 두 숫자 → [03](03_yolo_tensor.md) |
| **COCO 80 클래스** | 이 모델이 구분하는 객체 종류 80가지(사람·차·신호등 등) → [03](03_yolo_tensor.md) |
| **mAP** | 검출 모델의 표준 정확도 지표. **이 프로젝트에서는 쓰지 않는다**(정답 라벨이 없어 경로 간 일치도로 판정) → [08](08_method.md) |

---

## G. Hailo NPU

| 용어 | 뜻 |
|---|---|
| **HailoRT** | Hailo NPU를 쓰는 런타임 라이브러리. 이 프로젝트는 4.24.0 → [04](04_npu_hailort.md) |
| **HEF** | Hailo Executable Format. 모델을 그 칩 전용으로 미리 컴파일해 둔 파일 → [04](04_npu_hailort.md) |
| **VDevice** | HailoRT가 보는 가상 장치 객체 → [04](04_npu_hailort.md) |
| **InferModel** | 고수준 추론 API. 편하지만 **원시 출력에는 쓸 수 없다** → [04](04_npu_hailort.md) |
| **`infer_async`** | 저수준 비동기 추론 호출. 완료되면 콜백이 온다 → [04](04_npu_hailort.md) |
| **컨텍스트 (context)** | 큰 모델을 칩에 나눠 올릴 때의 분할 단위. 이 모델은 4개 → [04](04_npu_hailort.md) |
| **batch** | NPU에 한 번에 묶어 보내는 프레임 수 → [04](04_npu_hailort.md) |
| **slots** | 파이프라인이 동시에 붙잡는 프레임 수. **지연을 지배하는 변수** → [04](04_npu_hailort.md), [10](10_why_this_design.md) |
| **백프레셔 (backpressure)** | 너무 많이 밀어 넣지 않도록 막는 장치 → [04](04_npu_hailort.md), [10](10_why_this_design.md) |
| **`hw_frame_size`** | 장치가 기대하는 한 프레임의 바이트 수. `infer_async`에 이 값을 줘야 한다 → [04](04_npu_hailort.md) |
| **H2D / D2H** | Host to Device / Device to Host. 데이터 전송 방향 → [04](04_npu_hailort.md) |
| **VStream** | HailoRT의 고수준 스트림 추상화. 호스트 쪽 변환이 여기 붙는다 → [04](04_npu_hailort.md) |
| **ROUND_ROBIN** | VDevice 스케줄링 방식. 이걸 켜지 않으면 추론이 실패한다 → [04](04_npu_hailort.md) |
| **DFC (Dataflow Compiler)** | ONNX를 HEF로 바꾸는 Hailo 컴파일러. x86 전용이라 Pi에서는 돌릴 수 없다 → [12](12_hailo8_operator_support.md) |
| **HN / HAR** | DFC가 파싱한 Hailo 네트워크 표현(HN)과 그것을 담은 아카이브(HAR). 양자화·컴파일의 입력 → [12](12_hailo8_operator_support.md) |
| **alls** | DFC 모델 스크립트. 정밀도(`a8_w8` 등), NMS 위치, 입력 변환 같은 컴파일 지시를 적는다 → [12](12_hailo8_operator_support.md) |
| **nn_core** | Hailo 칩의 신경망 연산부. "칩에서 돈다"는 여기에 배치된다는 뜻 → [12](12_hailo8_operator_support.md) |
| **a8_w8 / a8_w4 / a16_w16** | 레이어 정밀도 표기. a = 활성값, w = 가중치 비트 수 → [12](12_hailo8_operator_support.md) |
| **softmax chain** | 칩이 softmax를 reduce_max → 빼기·exp → reduce_sum → 곱으로 쪼개 구현한 사슬. 이 안의 레이어는 16 bit가 안 된다 → [12](12_hailo8_operator_support.md) |

---

## H. 이 프로젝트 고유의 말

| 용어 | 뜻 |
|---|---|
| **슬롯 (slot)** | GPU·NPU·표시가 공유하는 프레임 자리 번호. **반납 조건이 둘**이다 → [09](09_code_map.md) |
| **프로브 (probe)** | 본체에 넣기 전에 "이게 되는가"를 가르는 별도 실행 파일 → [08](08_method.md), [09](09_code_map.md) |
| **관문 (gate)** | 프로브가 통과해야 하는 판정 기준(P1, F0, G5 등) → [08](08_method.md) |
| **기준 구현 (CPU reference)** | `01_src/ref/` 의 CPU 코드. **수식의 단일 진실 원천** → [09](09_code_map.md) |
| **정본** | 어떤 수치의 공식 출처 문서. 이 프로젝트에서는 `07_results/*/summary.md` → [08](08_method.md) |
| **신뢰 불가** | 측정 중 온도·클럭 경고가 떠서 성능 수치를 쓸 수 없다는 표시 → [08](08_method.md) |
| **인용 금지** | 결함 상태이거나 왜곡된 것이 확인돼 쓰면 안 되는 수치 → [08](08_method.md) |
| **보호대 (guard band)** | GPU NMS에서 IoU가 임계값에 너무 가까워 CPU와 판정이 갈릴 수 있는 좁은 구간(0.7 ± 4e-6). 걸린 프레임은 CPU로 처리한다 → [03](03_yolo_tensor.md) |
| **`yvo_`** | 이 프로젝트 빌드 타깃 이름의 접두사 (yolo vulkan optimize) → [09](09_code_map.md) |

---

## 다음에 읽을 것

### 이 프로젝트 안에서 — 이 순서를 권한다

| 순서 | 문서 | 왜 |
|---|---|---|
| 1 | [`README.md`](../README.md) | 실행 안내. §1 파이프라인 도식과 §4 옵션표 |
| 2 | [`07_results/README.md`](../07_results/README.md) | **수치를 인용하기 전에 여기부터.** 실험 25개 색인과 "설계를 지배하는 사실 6개" |
| 3 | [`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §8 | 최종 결론. "지금 뭘 쓰면 되나" |
| 4 | [`07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3 | "왜 이 설계인가"의 핵심 |
| 5 | [`09_diagrams/`](../09_diagrams/) | yEd로 여는 파이프라인 다이어그램 2장. 노드마다 소스 경로와 수치 출처가 붙어 있다 |
| 6 | [`10_paper/paper_ko.md`](../10_paper/paper_ko.md) | 논문 형식의 전체 서술. **이 문서들을 읽었다면 이제 읽을 수 있다** |

> ⚠ 논문은 본문은 완성됐지만 저자 정보와 참고문헌이 아직 채워지지 않았다.

### 이 프로젝트 밖에서

용어를 더 깊이 보고 싶을 때의 출발점이다. 이 프로젝트는 아래 중 어느 것도 요구하지 않는다 —
**필요한 것은 각 챕터에 다 들어 있다.**

| 주제 | 어디를 보면 되나 |
|---|---|
| Vulkan 스펙 원문 | 이 머신의 `/usr/share/vulkan/registry/` (`vk.xml`, `validusage.json`) |
| Vulkan 개념 설명 | Khronos 공식 Vulkan Guide / Vulkan Tutorial |
| dma-buf | 리눅스 커널 문서 `Documentation/driver-api/dma-buf.rst` |
| HailoRT API | Hailo Developer Zone 문서 (계정 필요) |
| YOLO 계열 모델 | Ultralytics YOLOv8/v11 문서 |
| 색공간·크로마 | ITU-R BT.709 권고 |

---

**처음으로** → [00_README.md](00_README.md)
