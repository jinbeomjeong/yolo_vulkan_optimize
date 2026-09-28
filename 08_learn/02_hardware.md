# 02. 하드웨어 — 이 보드 안에는 무엇이 들어 있는가

> **선수 지식**: [01_big_picture.md](01_big_picture.md)
> **다 읽으면 읽을 수 있는 것**: [`07_results/npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md),
> [`07_results/decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md),
> [`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3 (배경: 대상 플랫폼과 지배적 제약),
> [`07_results/display_e2e/env_bench.txt`](../07_results/display_e2e/env_bench.txt) 같은 환경 스냅샷 파일.

1장에서 "칩 다섯 개"가 각자 일을 나눠 맡는다고 했다. 이 장은 **그 칩들이 각각 무엇이고,
왜 그 일을 잘하고, 이 프로젝트에서 실제로 어떻게 판명됐는지**를 다룬다.
용어는 1장에서 쓴 것을 그대로 쓴다 — 처리량, 지연, glass-to-glass, 병목, 슬롯, zero-copy.

---

## 2.1 SoC — "한 칩 안에 컴퓨터가 다 들어 있다"

### 개념

PC를 열어 보면 큰 부품이 따로따로 꽂혀 있다. CPU 칩, 그래픽 카드, 메모리 컨트롤러가
각각 메인보드 배선으로 이어진다.

**SoC(System on a Chip, 시스템 온 칩)** 는 이것들을 **한 개의 칩 안에 밀어 넣은 것**이다.
휴대폰, 라즈베리파이처럼 작고 전기를 적게 써야 하는 기기는 거의 전부 SoC를 쓴다.
칩 안의 각 기능 덩어리를 **블록(block)** 이라고 부른다.

한 칩 안에 넣으면 두 가지가 좋아진다. (1) **거리가 짧다** — 블록 사이 배선이 수 밀리미터라
신호가 빨리 오가고 전기를 덜 쓴다. (2) **메모리를 같이 쓴다** — 칩 안의 모든 블록이
**같은 DRAM 하나**에 붙는다. 이게 뒤에 나올 zero-copy의 전제다
(→ [2.8](#28-메모리--통합-메모리와-캐시)).

### 이 보드의 SoC: BCM2712

이 프로젝트가 도는 보드는 **Raspberry Pi Compute Module 5 (CM5)**, 그 위에 얹힌 SoC가
**Broadcom BCM2712** 다 ([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3.1,
[`isp_resize/summary.md`](../07_results/isp_resize/summary.md) 1부). 안에 든 블록 중
이 프로젝트가 쓰는 것은 넷이다.

```
┌──────────────── BCM2712 (SoC) ─────────────────┐
│                                                 │
│  CPU          GPU          비디오       ISP     │
│  Cortex-A76   VideoCore    디코더      PiSP BE  │
│  × 4코어      VII (V3D)    (HEVC)               │
│    │           │            │           │       │
│    └───────────┴────────────┴───────────┘       │
│                   │                             │
│            메모리 컨트롤러                       │
└───────────────────┼─────────────────────────────┘
                    │                    ┌───────────┐
                 ┌──┴───┐   PCIe Gen3 x1 │  Hailo-8  │
                 │ DRAM │◄──────────────►│   (NPU)   │
                 └──────┘                └───────────┘
                                          ← SoC 밖의 별도 칩
```

**중요한 것은 Hailo-8이 이 상자 밖에 있다는 점이다.** NPU는 SoC의 일부가 아니라
**PCIe라는 통로로 따로 연결된 별개의 칩**이고, 이 사실이 이 프로젝트의 병목을 만든다.

---

## 2.2 CPU — 순서대로 시키는 대로 하는 일꾼

### 코어(core)와 클럭(clock)

**CPU(Central Processing Unit, 중앙처리장치)** 는 프로그램의 명령을 하나씩 읽어서 실행한다.
**코어**는 "그 일을 하는 일꾼 하나"다. 코어가 4개면 서로 다른 네 가지 일을 진짜로 동시에 할 수 있다.
이 보드의 CPU는 **ARM Cortex-A76 ×4** 다 — 일꾼 4명이다.

CPU는 일정한 박자에 맞춰 움직인다. 이 박자가 **클럭**이고 단위는 Hz(1초에 몇 번).
**2.4 GHz** = 1초에 24억 번 박자를 친다는 뜻이고, 박자가 빠르면 같은 시간에 더 많은 명령을 처리한다.
이 보드의 측정 시점 클럭이 **2.4 GHz**다
([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3.1,
[`07_results/display_e2e/env_bench.txt`](../07_results/display_e2e/env_bench.txt) 의 `frequency(0)=2400033792`).

**클럭은 고정된 값이 아니다.** 한가하면 내려가고 뜨거우면 강제로 내려간다 (→ [2.9](#29-열thermal과-클럭-강등)).

### 캐시(cache)란 무엇인가 — 그리고 왜 계층인가

CPU가 DRAM(주기억장치)에서 데이터를 가져오는 데는 **CPU 박자 기준으로 수백 박자**가 걸린다.
매번 그러면 일꾼이 대부분의 시간을 기다리며 논다. 그래서 CPU 바로 옆에 **작지만 아주 빠른
임시 저장소**를 둔다. 이것이 **캐시**다 — 한 번 읽은 데이터를 남겨 두고, 다음에 같은 것을 찾으면
DRAM까지 가지 않고 캐시에서 준다. 작고 빠른 것과 크고 느린 것을 **여러 층**으로 쌓기 때문에
**캐시 계층(hierarchy)** 이라고 한다 (아래는 `/sys/devices/system/cpu/cpu0/cache/` 에서 읽은 값):

| 층 | 크기 | 누가 쓰나 | 성격 |
|---|---|---|---|
| **L1** | 데이터 64 KiB + 명령 64 KiB | **코어 하나마다 따로** | 가장 작고 가장 빠르다 |
| **L2** | 512 KiB | 코어 하나마다 따로 | 중간 |
| **L3** | 2 MiB | **4코어가 공유** | 가장 크고 가장 느리다(그래도 DRAM보다 훨씬 빠르다) |

> **감을 잡는 숫자 하나.** 1920×1080 NV12 프레임 한 장은 약 3 MB다
> (1장 §1.5). **L3 캐시 2 MiB보다 크다.** 즉 프레임 한 장을 CPU로 처음부터 끝까지
> 훑으면 캐시에 담기지 않아 DRAM을 통째로 왕복한다. "CPU로 프레임을 만지는 일"이
> 비싼 근본 이유가 이것이다.

### "CPU 시간 2.18 ms/frame"은 코어 몇 개분인가

1장에서 최종 결과로 **CPU 2.18 ms/frame**을 봤다
([`07_results/display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3, `disp_source`).
이 숫자가 얼마나 작은 건지 계산해 보자.

- `disp_source` 실행은 **29.91 FPS**, 프레임당 CPU **2.179 ms**
- 1초 동안 쓰는 CPU 시간 = 29.91 × 2.179 ms ≈ **65 ms**
- 코어 하나는 1초에 1000 ms를 낸다 → **한 코어의 약 6.5%**, 4코어 기준 **약 1.6%**

`disp_asap` 실행(34.87 FPS, 1.842 ms/frame)으로 계산해도 34.87 × 1.842 ≈ 64 ms로 거의 같다.

> ⚠ 두 실행은 **다른 조건**이라 FPS·CPU·glass 값을 서로 섞어 인용하면 안 된다
> ([`07_results/README.md`](../07_results/README.md) 맨 위 경고). 위에서는 각 실행 안의
> 값끼리만 곱했기 때문에 문제가 없다.

**비교 대상이 있어야 의미가 산다.** 같은 1080p 영상을 CPU만으로 디코드하고 640×640 RGB로
변환하면 프레임당 약 28 ms가 든다
([`decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md), `avdec_h264` S2).

| 경로 | CPU ms/frame | 30 FPS에서 쓰는 코어 |
|---|---:|---|
| CPU 디코드 + CPU 변환 | 28.1 | 약 0.84 코어 (21%) |
| **이 프로젝트 최종(표시 포함)** | **2.18** | **약 0.065 코어 (1.6%)** |

**약 13배 차이다.** 이 프로젝트가 한 일의 본질이 이 표 한 줄이다.

---

## 2.3 GPU — 같은 일을 수백만 번 반복하는 데 특화된 칩

### 왜 GPU가 대량 반복에 강한가

CPU 코어는 **똑똑하지만 소수**다. "이 값이 3보다 크면 A를, 아니면 B를" 같은 복잡한 조건 분기를
빠르게 처리하도록 설계돼 있고, 그러려면 코어 하나가 크고 비싸진다. 그래서 4개뿐이다.
**GPU(Graphics Processing Unit, 그래픽 처리 장치)** 는 반대다. 코어 하나하나는 단순하지만
**아주 많고**, 결정적으로 **여러 데이터에 같은 명령을 한꺼번에 먹인다.** 이 방식을
**SIMD**(Single Instruction, Multiple Data — 명령 하나, 데이터 여럿)라고 한다.
"이 픽셀의 밝기를 1.2배 해라"라는 명령 하나를 16개 픽셀에 동시에 먹이는 식이다.

영상 처리는 이 방식에 완벽하게 맞는다. 1920×1080 = 207만 픽셀 전부에 **똑같은 계산**
(YUV를 RGB로 바꾸기, 크기 줄이기)을 하고, 조건 분기가 거의 없으며,
픽셀끼리 서로 결과를 참조하지도 않기 때문이다.

### 이 보드의 GPU: VideoCore VII (V3D 7.1)

칩 이름은 **VideoCore VII**, 하드웨어·드라이버 세대를 가리키는 이름은 **V3D 7.1**이다.
리눅스에서 이 GPU를 Vulkan으로 쓰게 해 주는 드라이버가 **Mesa V3DV 26.2.2**이고 지원 Vulkan 버전은
**1.3** 이다 ([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3.1).

SIMD의 폭과 한 번에 묶을 수 있는 일의 크기는 하드웨어마다 다르다. V3D 7.1은 이렇고
([`paper_ko.md`](../10_paper/paper_ko.md) §3.3 제약 3), 이 숫자들이 셰이더를 어떻게 짜야 하는지를
직접 제약한다.

- **서브그룹(subgroup) 크기 16 고정** — 한 명령이 동시에 먹는 데이터가 16개 묶음이다
- **워크그룹(workgroup) 최대 256 invocation** — 한 덩어리로 묶어 보낼 수 있는 작업 단위가 256개까지
- 스테이지당 스토리지 버퍼 8개, 스토리지 이미지 4개, `descriptorIndexing` 미지원

### 타일 기반 렌더러(tile-based renderer)란 무엇인가

데스크톱 GPU는 화면 전체를 담는 큰 메모리(프레임버퍼)에 곧장 그림을 그린다. 메모리 대역폭을
많이 먹지만 전기를 많이 써도 되는 환경이라 괜찮다.
**모바일·임베디드 GPU는 전기가 아깝다.** 그래서 화면을 작은 **타일(tile)** 로 잘라서,
타일 하나를 GPU 칩 안의 작은 초고속 메모리에 올려놓고 거기서 다 그린 다음
**완성된 타일만 DRAM에 한 번 내보낸다.** 이 방식이 **타일 기반 렌더링**이고 V3D가 이 방식이다.

DRAM 왕복이 줄어 전기를 크게 아끼지만 **대가가 있다.** 타일에 그림을 그리려면 먼저 그 타일의
현재 내용을 DRAM에서 **불러와야(load)** 하고 끝나면 **내보내야(store)** 한다. 이미 다른 방법으로
채워 놓은 이미지 위에 그래픽스 방식으로 조금만 덧그리려 해도
**프레임버퍼 전체의 load/store가 통째로 발생한다.**

**이 프로젝트가 여기서 설계를 바꿨다.** 검출 박스 오버레이를 보통대로
그래픽스 파이프라인(삼각형을 그리는 방식)으로 만들면 위 비용을 그대로 맞는다. 그래서 오버레이를
**컴퓨트 셰이더로 직접 픽셀을 찍는 방식**(컴퓨트 래스터라이저)으로 구현했다
([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3.3 제약 3 마지막 항목, 구현은 §4.5).
셰이더는 [`02_shaders/overlay_boxes.comp`](../02_shaders/overlay_boxes.comp) 이고 실측 0.128~0.196 ms다
([`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.1).

타일 구조는 성능 숫자에도 직접 나타난다. 표시 변환 셰이더의 워크그룹 모양을
16×16(64×16 픽셀 타일)에서 3×85(12×85 픽셀)로 바꿨더니 **7.74 ms → 4.23 ms로 1.83배 빨라졌다**
— 같은 invocation 수인데 세로로 긴 모양이 결정적으로 빠르다
([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 3).

> **가설이다.** 그 문서는 원인을 "타일 배치된 RGBA 이미지 쓰기 쪽이 병목이고 세로로 긴 영역이
> 이미지 타일에 더 잘 맞기 때문"으로 **추정**하며, **"이 해석은 미확정 가설이고 수치만이 근거다"**
> 라고 명시한다. 읽기(SAND 열이 가로 128 B)만 보면 오히려 불리해야 하기 때문이다.

상세는 [05_vulkan_compute.md](05_vulkan_compute.md).

---

## 2.4 NPU — 신경망 전용 칩, 그리고 그게 SoC 밖에 있다는 것

### NPU란 무엇인가

**NPU(Neural Processing Unit, 신경망 처리 장치)** 는 딥러닝 모델이 쓰는 연산
(주로 행렬 곱셈과 합성곱)만 아주 빠르고 전기를 적게 쓰며 하도록 만든 칩이다.
GPU로도 신경망을 돌릴 수 있지만, NPU는 그 한 가지만 하기 때문에 같은 전력에서 훨씬 빠르다.

이 프로젝트의 NPU는 **Hailo-8**이고, 돌리는 모델은 [`03_models/yolov11m.hef`](../03_models/) 다
(`.hef` = Hailo Executable Format). 입력은 640×640×3 UINT8,
출력은 원시 텐서 6개 1,221,120 바이트다
([`07_results/npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md), 1장 §1.4 ③).

### 결정적인 사실: Hailo-8은 SoC 안에 없다

GPU·비디오 디코더·ISP는 전부 BCM2712 **안**에 있다. 같은 칩 안이니 같은 DRAM을 바로 본다.

**Hailo-8은 다르다. 별개의 칩이고 PCIe 링크로 보드에 붙어 있다.**
그래서 프레임을 NPU에 넣으려면 **데이터를 PCIe 통로로 실어 보내고 결과도 되받아야** 한다.
프레임 한 장당 주고받는 양은 입력 640×640×3 ≈ 1.23 MB + 출력 1,221,120 B ≈ 1.22 MB =
**약 2.4 MB**다. 이 왕복 비용이 이 프로젝트의 병목이다 — 다음 절이 그 증거다.

HailoRT(Hailo의 런타임 라이브러리)를 어떻게 쓰는지는 [04_npu_hailort.md](04_npu_hailort.md).

---

## 2.5 PCIe — 칩과 칩을 잇는 고속 통로

### 개념: 세대와 레인

**PCIe(Peripheral Component Interconnect Express)** 는 컴퓨터 안에서 칩·카드를 잇는
표준 고속 통로다. PC의 그래픽 카드, NVMe SSD가 꽂히는 그 규격이고, 속도는 둘이 정한다.

1. **세대(generation)** — 통로 하나의 신호 속도. 단위는 **GT/s**(Giga Transfers per second).
   **Gen2 = 5.0 GT/s, Gen3 = 8.0 GT/s.** (GT/s가 그대로 바이트 전송량은 아니다. 신호에는
   오류 검출용 여분 비트가 섞여 있고 그 비율이 세대마다 달라서, 실효 대역폭은
   Gen2 → Gen3에서 **대략 2배**가 된다.)
2. **레인(lane) 수** — 통로를 몇 가닥 병렬로 깔았는가. `x1`은 한 가닥, `x4`는 네 가닥이고
   가닥 수에 비례해 대역폭이 는다.

**이 보드는 Gen3 x1** — **가장 좁은 구성인 한 가닥**이고 세대만 최신이다
([`display_e2e/env_bench.txt`](../07_results/display_e2e/env_bench.txt):
`pcie: 8.0 GT/s PCIe x1 (Gen3, 0001:01:00.0)`).

### 현재 세대를 확인하는 법 / 바꾸는 법

세대는 **문서를 믿지 말고 읽어야 한다.** 이 프로젝트의 측정 프로토콜이 그렇게 못박고 있다
([`.claude/skills/measure/SKILL.md`](../.claude/skills/measure/SKILL.md) §1 —
"과거에 에이전트 정의에 적혀 있던 낡은 값을 그대로 인용해, Gen3에서 잰 측정을 Gen2로 보고한 적이 있다").
읽는 곳은 `/sys/bus/pci/devices/0001:01:00.0/current_link_speed`
(`5.0 GT/s` = Gen2, `8.0 GT/s` = Gen3), 바꾸는 곳은 `/boot/firmware/config.txt` 의
`dtparam=pciex1_gen` 이다
([`npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md)).

### 측정: 세대를 바꿨더니 NPU 처리량이 거의 2배가 됐다

이 프로젝트는 **같은 보드, 같은 모델, 같은 명령으로 PCIe 세대만 바꿔** 두 번 쟀다
([`npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md)).

| batch | Gen2 x1 | Gen3 x1 | 배율 |
|---:|---:|---:|---:|
| 1 | 12.71 | 24.34 | 1.92× |
| 2 | 18.09 | 34.93 | 1.93× |
| 4 | 22.62 | 44.52 | 1.97× |
| **8** | 26.07 | **51.71** | **1.98×** |
| 16 | 22.39 | 43.72 | 1.95× |

(`batch`는 "한 번에 몇 장을 묶어서 NPU에 보내는가"다. 상세는 [04_npu_hailort.md](04_npu_hailort.md).)

### 왜 이것으로 병목이 "확정"되는가

배율이 1.9~2.0배로 나온 것만으로는 부족하다. 더 강한 증거는 **시간을 분해했을 때**다.
한 배치를 처리하는 시간 `T`를 "배치당 고정비 + 프레임당 비용"으로 나누면
([`paper_ko.md`](../10_paper/paper_ko.md) §3.3 제약 2,
[`latency_gen3_cooled/summary.md`](../07_results/op_point/runs/latency_gen3_cooled/summary.md) 결론 3):

| | 배치당 고정비 | 프레임당 |
|---|---:|---:|
| Gen2 x1 | 약 46 ms | 32.6 ms |
| **Gen3 x1** | **약 24.8 ms** | **16.2 ms** |

**두 항이 모두 정확히 반으로 줄었다.** 논리는 이렇다. 시간의 일부가
"NPU 칩 안에서 계산하는 시간"이었다면 PCIe 세대를 바꿔도 그 부분은 그대로 남아야 한다
(계산 속도는 링크와 무관하므로). 그런데 **고정비도 프레임당 비용도 둘 다 절반이 됐다.**
→ 시간의 거의 전부가 **통로를 건너는 데 쓰이고 있다**는 뜻이다.

Gen3의 선형 모델은 `T(B) = 24.8 + 16.2·B [ms]` 이고,
여기서 나오는 점근 처리량 상한은 `1 / 0.0162 ≈ 61.6 FPS`, 실측 상한은 **51.7 FPS**다.

> **여기까지가 확정이고, 그 아래는 가설이다.** "전송이 시간을 지배한다"는 위 증거로 확정됐다.
> 다만 그것이 **순수한 대역폭 포화**인지, 아니면 전송 한 번마다 붙는 **고정 오버헤드**인지는
> 이 프로젝트가 더 파고들지 않았다. 논문도 "두 항이 모두 전송 지배적임을 **시사한다**"라고만 쓴다.

**결론: 앞뒤 단계를 아무리 최적화해도 51.7 FPS를 넘을 수 없다.**
1장에서 "FPS를 올리는 것을 목표로 하지 않았다"고 한 이유가 이 표다.

---

## 2.6 비디오 디코더 — SoC 안의 전용 블록, 단 HEVC만

영상 파일 안의 그림은 **압축**돼 있다(1장 §1.4 ①). 압축을 푸는 것이 **디코드**다.
압축 방식(코덱)에는 여러 종류가 있고 이 프로젝트에 나오는 것은 둘이다 —
**H.264**(= AVC, 오래됐고 가장 널리 쓰인다)와
**H.265**(= HEVC, High Efficiency Video Coding, 더 새롭고 같은 화질에 파일이 더 작다).

디코드는 CPU가 소프트웨어로 할 수도 있고(느리고 CPU를 많이 먹는다),
**전용 하드웨어 블록**이 할 수도 있다(빠르고 CPU가 거의 놀 수 있다).

### 이 보드의 사실: HEVC만 하드웨어로 풀린다

**BCM2712에는 HEVC 하드웨어 디코더가 있지만 H.264 하드웨어 디코더는 없다.** 그래서 H.264 영상은
CPU가 소프트웨어로 풀 수밖에 없다. 추측이 아니라 측정으로 확인했다
([`decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md), 1080p 중앙값).

| 경로 | 코덱 | 처리량 | 프레임당 CPU |
|---|---|---:|---:|
| `avdec_h264` (소프트웨어, 4코어) | H.264 | 164.6 FPS | **20.2 ms** |
| `avdec_h265` (소프트웨어, 4코어) | HEVC | 109.1 FPS | **31.8 ms** |
| **ffmpeg `-hwaccel drm` (하드웨어, CLI)** | **HEVC** | **332.8 FPS** | **2.0 ms** |

하드웨어 디코더를 쓰면 CPU가 **2.0 ms**로 끝난다 — 소프트웨어의 10~16분의 1이다.
단 이 2.0 ms도 대부분(약 1.5 ms)은 CLI가 프레임을 CPU로 내려받아 변환하는 비용이다. DRM_PRIME을 유지하면 0.55 ms다
([`decode_benchmark/runs/ffmpeg_outfmt`](../07_results/decode_benchmark/runs/ffmpeg_outfmt/summary.md), 2026-09-29).
그래서 이 프로젝트의 기본 입력 영상이 [`04_data/videos/video_h265.mp4`](../04_data/videos/)
(HEVC 1080p30)다.

### 함정 하나 — "하드웨어를 쓰는데 왜 안 빠르지"

같은 하드웨어 디코더를 GStreamer의 `v4l2slh265dec`로 쓰면 **32.7 FPS, CPU 30.6 ms/frame**이 나왔다(2026-09-20, gst-plugins-bad rpt3).
하드웨어를 쓰는데도 소프트웨어 디코드와 CPU 비용이 거의 같다. 원인은 디코더가 아니라
**그 뒤의 복사**다. `perf` 프로파일에서 CPU 시간의 약 90%가 프레임 복사·변환 코드였다 —
디코더가 내놓는 **SAND 타일 포맷**을 CPU 한 스레드가 일반 메모리로 옮기고 있었다
(같은 문서 해석 1 — 문서는 이 원인을 "추정한다"고 적는다).
이 복사는 알려진 upstream 버그(1.26.4에서 수정)였고, 2026-09-29 rpt4 패키지가 수정을 들여오자 DMABuf 경로가
**381.5 FPS / 0.24 ms/frame**이 됐다([`decode_benchmark/runs/gst_rpt4`](../07_results/decode_benchmark/runs/gst_rpt4/summary.md)).

**교훈: 하드웨어 블록을 썼다는 사실만으로는 아무것도 보장되지 않는다.
그 출력이 다음 단계까지 복사 없이 가느냐가 실제 비용을 정한다.**

그래서 이 프로젝트는 GStreamer 대신 libavcodec의 DRM_PRIME 경로를 직접 써서 디코더 출력 dma-buf를
GPU가 바로 읽게 만들었다. 디코드 CPU 비용은 **0.23~0.28 ms/frame**까지 내려간다
([`07_results/README.md`](../07_results/README.md) 실험 목록, 원 출처는
[`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.1).
상세는 [07_video_display.md](07_video_display.md), zero-copy는 [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md).

---

## 2.7 ISP (PiSP Back End) — 원래 카메라용인데 크기도 바꿔 준다

**ISP(Image Signal Processor, 이미지 신호 프로세서)** 는 카메라 센서에서 나온 **날것의 신호**를
사람이 볼 만한 사진으로 다듬는 전용 블록이다. 화이트밸런스, 노이즈 제거, 렌즈 왜곡 보정,
색 보정 같은 일을 한다. BCM2712의 ISP는 **PiSP Back End(PiSP BE)** 라고 부른다.

**그런데 ISP는 크기 변환(resize)과 색 형식 변환도 한다.** 카메라 파이프라인에서 당연히 필요한
기능이기 때문이다. 그러면 이런 생각이 든다 — **"전처리(1920×1080 → 640×640 letterbox + RGB 변환)를
GPU 대신 ISP한테 시키면 어떨까?"**

### 이 프로젝트가 실제로 해 봤고, 보류했다

구현까지 끝냈고([`01_src/isp/`](../01_src/isp/)), 측정 결과는
[`isp_resize/summary.md`](../07_results/isp_resize/summary.md)에 1부(전처리 단독)와
2부(전체 파이프라인)로 나뉘어 있다. **얻은 것은 확실했다.**

| 항목 | GPU 셰이더 | ISP |
|---|---:|---:|
| 전처리 시간 (같은 wall-clock 축) | 2.081 ms | **1.594 ms (−23%)** |
| V3D(GPU) 점유 | 2.718 ms | **0** |

**대가도 확실했다.** CPU가 프레임당 **+0.89 ms**(C안) 또는 **+1.50 ms**(A안) 늘어난다.
FPS와 지연은 전혀 변하지 않는다 — 병목이 NPU라 여기서 아껴도 소용이 없기 때문이다
(세 설정 모두 51.6 FPS로 동일). **그리고 검출이 막았다.**

| 지표 | 합격선 | 측정값 | 판정 |
|---|---|---|---|
| 총 검출 수 | 9457 ± 1% | **9237 (−2.33%)** | **불합격** |
| 매칭 비율 @ IoU 0.5 | ≥ 0.99 | **0.7940** | **불합격** |

원인을 파고든 결과 **박스 위치가 틀린 게 아니었다.** 점수 0.4 이상인 검출은 양쪽에서 99.5% 이상
똑같이 재현되고, 흔들린 것은 점수 0.2~0.3 구간의 **경계선 검출**뿐이다. ISP 출력이 CPU 기준보다
평균 **−0.5 LSB** 어두워서 그 구간의 신뢰도가 내려간 것으로 **보인다**(문서도 "~로 보인다"라고 쓴다
— **확정된 원인이 아니라 가설이다**).

**판정: 보류.** 정답 라벨이 없어서 9237과 9457 중 어느 쪽이 더 맞는지 판정할 수 없고,
지금은 V3D가 남아도는 상태라 "GPU 2.1 ms를 아끼려고 CPU 0.9 ms를 쓰는" 교환이 남는 장사가 아니다.
같은 문서가 재검토 조건 3개를 명시해 뒀다(V3D가 실제로 부족해질 때 / 정답 라벨로 판정된 뒤 /
CSI 카메라 입력으로 갈 때). → 1장에서 말한 **"빠르다고 늘 옳은 게 아니다"** 의 실물 사례다.

---

## 2.8 메모리 — 통합 메모리와 캐시

### 통합 메모리(unified memory)가 zero-copy의 전제다

데스크톱 PC의 그래픽 카드는 **자기만의 메모리(VRAM)** 를 따로 들고 있다.
CPU 쪽 메모리에 있는 데이터를 GPU가 쓰려면 PCIe로 **복사해 넣어야** 한다. 피할 수 없다.

**SoC는 다르다.** CPU·GPU·비디오 디코더·ISP가 **같은 DRAM 하나**를 보고, 이것을
**통합 메모리**라고 한다. 그래서 이런 게 가능해진다 — 디코더가 DRAM 어딘가에 프레임을 써 놓고
**그 위치를 GPU에게 알려주기만 하면** GPU가 그 자리를 직접 읽는다. 복사가 0이다.
이것이 1장에서 말한 **zero-copy**이고, "위치를 알려주는 수단"이 **dma-buf**다.

**통합 메모리가 아니면 zero-copy는 애초에 성립하지 않는다.** Hailo-8은 SoC 밖에 있지만,
PCIe 장치가 시스템 DRAM을 직접 읽고 쓰는 **DMA**(Direct Memory Access, CPU를 거치지 않고
장치가 직접 메모리를 접근하는 것)가 가능해서 같은 dma-buf를 NPU에도 넘길 수 있다.

### 그런데 "같은 메모리"라고 성능까지 같지는 않다 — 캐시 매핑의 차이

문제는 **같은 DRAM 영역이라도 어떤 방식으로 매핑(mapping, CPU가 그 주소를 보는 방식)됐느냐에 따라
캐시를 쓸 수도, 못 쓸 수도 있다**는 것이다.

| 매핑 종류 | CPU 읽기 | CPU 쓰기 | 언제 쓰나 |
|---|---|---|---|
| **cached** (캐시 사용) | 빠르다 (캐시에서 준다) | 빠르다 | 보통의 메모리 |
| **uncached** (캐시 안 씀) | **매우 느리다** (매번 DRAM까지) | 느리다 | 장치와 공유하는 영역 |
| **write-combine** | **느리다** (매번 DRAM까지) | **빠르다** (모았다가 한 번에) | GPU가 소비하는 영역 |

**핵심은 write-combine의 비대칭이다.** 쓰기는 모았다가 한 번에 내보내서 빠르지만
읽기는 캐시가 없으니 매번 DRAM까지 간다.

### 이 보드에서의 구체적 사실

V3D 드라이버가 내놓는 Vulkan 메모리 타입은 **`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT` 한 종류뿐**이고
**`HOST_CACHED`가 없다.** 즉 **GPU가 쓴 버퍼를 CPU가 캐시된 상태로 읽는 선택지가 드라이버 수준에 아예 없다**
([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3.3 제약 1). 실측된 비대칭은 이렇다.

| 측정 | 값 | 출처 |
|---|---:|---|
| GPU가 쓴 메모리를 CPU가 **순차로** 읽기 | **1.479 GB/s** (0.826 ms / 1,221,120 B) | [`gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md) P2 ⑥ |
| 같은 양을 **캐시 매핑 dma-heap**에서 읽기 | 3.084 GB/s (0.396 ms) | 〃 |
| 같은 양을 **평범한 malloc**에서 읽기 | 5.001 GB/s (0.244 ms) | 〃 |
| GPU가 쓴 메모리를 CPU가 **stride-3으로** 읽기 | **약 0.088 GB/s** (약 14 ms) | [`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3 |
| 반대 방향(CPU → GPU 쓰기) | **싸다** | [`07_results/README.md`](../07_results/README.md) 2번 |

> ⚠ **순차 읽기에는 서로 다른 두 측정이 있다.** 위 표의 1.479는 할당처 세 가지를 나란히 잰
> P2 프로브 값이고(30회 반복), 파이프라인 버퍼를 잰 `yvo_preproc_bench` 쪽은 1,228,800 B에
> 0.85~0.87 ms = **약 1.41 GB/s** 다. **둘을 같은 값처럼 섞지 마라.**
> 할당처 비교에는 P2를 쓴다 — [`07_results/README.md`](../07_results/README.md) 2번 참조.
| 1.2 MB memcpy — 캐시 매핑 dma-heap(ISP 출력) | **0.229 ms** | [`isp_resize`](../07_results/isp_resize/summary.md) 1부 표 2 |
| 1.2 MB memcpy — V3D 메모리(GPU 출력) | **1.045 ms** (**4.6배**) | 〃 |

**이 한 가지 사실이 이 프로젝트의 설계를 통째로 지배한다.**
"GPU가 쓴 버퍼를 CPU로 훑는 모든 작업은 겉보기보다 약 10배 비싸다"가 그 결론이고,
NPU 입력 레이아웃을 NHCW로 바꿔 CPU가 그 메모리를 읽는 일 자체를 없앤 것이
**CPU 17.6 → 3.5 ms/frame** 개선의 본질이다
([`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3).
**여기서는 개념만 잡는다** — dma-buf, dma-heap, 캐시 동기화(`DMA_BUF_IOCTL_SYNC`),
NHCW 레이아웃의 상세는 → [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md).

---

## 2.9 열(thermal)과 클럭 강등

칩은 일을 하면 열이 난다. 너무 뜨거워지면 망가지므로, 칩은 스스로를 지키려고
**클럭을 강제로 낮춘다.** 이것을 **스로틀링(throttling)** 또는 **클럭 강등**이라고 한다.

**성능 측정 입장에서 이건 재앙이다.** 코드는 한 줄도 안 바뀌었는데 "아까는 100, 지금은 93"이 나오고,
**대개 아무 에러도 경고도 없이 조용히 그렇게 된다.**

실제로 Hailo-8 칩은 연속 부하에서 **최고 92.7 °C**까지 올랐고, 그러면 칩 클럭이
**400 MHz → 250 MHz**로 떨어진다. 이때 런타임 로그에
`HEF was compiled assuming clock rate of 400 MHz, while the device clock rate is 250 MHz`
경고가 찍힌다 ([`npu_benchmark/runs/gen3/summary.md`](../07_results/npu_benchmark/runs/gen3/summary.md)).

### 그래서 이 프로젝트는 측정 전에 냉각 게이트를 둔다

규칙은 [`.claude/skills/measure/SKILL.md`](../.claude/skills/measure/SKILL.md) §2에 있다.

- 실행 직전 칩 온도가 **75 °C를 넘으면 65 °C 이하가 될 때까지 기다린다**(최대 120초)
- **직전 실행의 로그에 온도·클럭 경고가 있으면 무조건 냉각한다**
- 냉각 대기 시간도 기록한다 — 측정 전체 소요의 절반 이상일 수 있다
  (실제로 [`display_e2e`](../07_results/display_e2e/summary.md) 측정에서는 합계 **315초**였다)
- 로그에 온도·클럭 경고가 **1건이라도** 있으면 그 실행의 수치를 **"신뢰 불가"로 표시한다**

### 과열 측정이 실제로 얼마나 틀렸는가 — 숫자로

이 프로젝트에는 과열 상태에서 잰 탓에 **버려진 측정**이 있다. 냉각 상태에서 다시 재서 차이를 직접
보여 준 문서가 [`latency_gen3_cooled/summary.md`](../07_results/op_point/runs/latency_gen3_cooled/summary.md) 다.

| batch | 항목 | 과열 (신뢰 불가) | **냉각 (재측정)** | 차이 |
|---:|---|---:|---:|---:|
| 1 | HW 지연 | 34.62 ms | **32.513 ms** | **−6.1%** |
| 1 | Overall 지연 | 42.78 ms | **40.894 ms** | **−4.4%** |
| 8 | HW 지연 | 133.35 ms | **126.390 ms** | **−5.2%** |
| 8 | Overall 지연 | 155.75 ms | **144.863 ms** | **−7.0%** |

**과열 상태 측정은 지연을 4~7% 부풀렸다.** 방향과 크기가 모두 일관된다. 덤으로 이 재측정이
묵은 의문 하나도 풀었다 — 두 측정 도구가 다른 지연 값을 내서 "도구가 다른가?" 싶었는데
냉각 상태에서 재니 **사실상 같은 값**이었다(차이 0.01% / 0.05%). 도구 차이가 아니라
**한쪽이 뒤에 실행돼서 더 뜨거웠던 것**이다.

### 열만 클럭을 바꾸는 게 아니다 — 거버너

리눅스는 **거버너(governor)** 라는 정책으로 "한가하면 클럭을 낮춰 전기를 아낀다". 그래서
**일을 덜 하는 설정이 오히려 프레임당 ms가 더 크게 나오는** 역설이 생긴다. 실제로 겪었다 —
`disp_source`(29.91 FPS)의 CPU 2.179 ms가 `disp_asap`(34.87 FPS)의 1.842 ms보다 큰데 일은 덜 한다.
페이싱 때문에 프로세스가 놀아서 거버너가 클럭을 내린 것으로 **보인다**
([`07_results/display_e2e/summary.md`](../07_results/display_e2e/summary.md) §7 — 문서도 "보인다"로
쓴다. **확정된 원인이 아니다.**) **교훈: CPU 시간(ms)을 비교할 때는 그때의 클럭을 함께 봐야 한다.**

---

## 2.10 이 프로젝트에서는 — 누가 무엇을 맡고, 누가 남아도는가

### 칩별 담당 (1장 §1.3 표를 더 자세히)

| 칩 | 위치 | 맡은 단계 | 코드 |
|---|---|---|---|
| **CPU** Cortex-A76 ×4 @2.4 GHz | SoC 안 | 전체 지휘, 슬롯 수명 관리 (NMS는 GPU, 드문 경계 프레임만 CPU) | [`01_src/pipeline/`](../01_src/pipeline/) |
| **비디오 디코더** (HEVC 전용) | SoC 안 | ① mp4 → 프레임 (dma-buf) | libavcodec DRM_PRIME |
| **GPU** VideoCore VII / V3D 7.1 | SoC 안 | ② 전처리, ④ 후처리, ⑥ 표시 변환, ⑦ 오버레이 | [`02_shaders/`](../02_shaders/) |
| **NPU** Hailo-8 | **SoC 밖, PCIe Gen3 x1** | ③ YOLO 추론 | [`01_src/hailo/`](../01_src/hailo/) |
| **ISP** PiSP BE | SoC 안 | (전처리 대체 경로 — **구현됐으나 미사용**) | [`01_src/isp/`](../01_src/isp/) |

### 어느 칩이 포화이고 어느 칩이 남아도는가

측정 출처는 [`display_e2e/summary.md`](../07_results/display_e2e/summary.md),
운용점은 `--batch 2 --slots 4`, 표시 전체화면이다.

#### NPU — **포화. 병목이다.**

파이프라인 전체 FPS가 **34.960**인데(§3 `reg_host_off`), 같은 batch 2로 NPU **단독**을 잰 값이
**34.926 FPS**다
([`op_point/runs/latency_gen3_cooled/summary.md`](../07_results/op_point/runs/latency_gen3_cooled/summary.md) §2).
**차이가 0.1%다.** 파이프라인이 NPU 단독 상한과 사실상 같은 속도로 돈다 —
즉 **다른 모든 단계가 NPU를 기다리고 있다.** 이보다 명확한 병목 증거는 없다.

#### GPU — **여유 있다. 주기의 약 24.5%.**

`disp_asap` 실행에서 GPU 4패스의 중앙값 합이다.

| 구간 | median | p90 |
|---|---:|---:|
| ② 전처리 `sand_letterbox.comp` | 2.088 ms | 2.205 |
| ④ 후처리 `yolo_decode.comp` | 0.626 ms | 0.704 |
| ⑥ 표시 변환 `display_convert.comp` | 4.153 ms | 4.207 |
| ⑦ 오버레이 `overlay_boxes.comp` | 0.128 ms | 0.154 |
| **합** | **6.995 ms** | **7.270 ms** |

이 실행의 프레임 주기는 28.6 ms다 → **median 24.5%, p90 25.4%.**
**GPU는 프레임 예산의 4분의 1만 쓰고 4분의 3이 비어 있다.** 게다가 NPU가 동시에 돌아도
느려지지 않는다 — NPU 유휴 기준선과 비교하면 네 구간 전부 같거나 오히려 빠르다
(최대 −34.7%, 최소 −0.2%). **"NPU 동시 실행에 의한 GPU 저하는 관측되지 않았다."**

#### CPU — **거의 비어 있다.** / ISP — **놀고 있다.**

CPU는 [2.2](#22-cpu--순서대로-시키는-대로-하는-일꾼)에서 계산한 대로 **한 코어의 약 6.5%,
4코어 전체의 약 1.6%** 다(표시를 끄면 1.115 ms/frame까지 더 내려간다). ISP는 구현은 돼 있지만
쓰지 않는다 ([2.7](#27-isp-pisp-back-end--원래-카메라용인데-크기도-바꿔-준다)).

### 그래서 이 하드웨어 지도가 말하는 것

```
NPU  ████████████████████████  포화 (병목)   GPU  ██████░░░░░░░░░░░░░░░░░░  24.5%
CPU  █░░░░░░░░░░░░░░░░░░░░░░░  ~1.6%        ISP  ░░░░░░░░░░░░░░░░░░░░░░░░  미사용
```

**병목은 PCIe로 붙은 NPU 한 곳이고, 나머지는 전부 남는다.**
그래서 이 프로젝트의 목표가 "FPS 올리기"가 아니라 **"남는 칩으로 CPU를 대신 일하게 해서
CPU를 비우기"** 가 된 것이다. 이 판단의 사슬은 [10_why_this_design.md](10_why_this_design.md).

---

## 2.11 실제 환경 사양 표

측정 때 실제로 기록된 값만 적는다. 확인하지 못한 항목은 비워 뒀다.

| 항목 | 값 | 출처 |
|---|---|---|
| 보드 | Raspberry Pi Compute Module 5 Rev 1.0 | [`isp_resize/summary.md`](../07_results/isp_resize/summary.md) 1부 |
| SoC | Broadcom BCM2712 (`BCM2712_D0`, hw_revision `0x2252701`) | [`isp_resize/summary.md`](../07_results/isp_resize/summary.md) 1부 |
| CPU | ARM Cortex-A76 ×4 @ **2.4 GHz** | [`paper_ko.md`](../10_paper/paper_ko.md) §3.1, [`display_e2e/env_bench.txt`](../07_results/display_e2e/env_bench.txt) |
| CPU 캐시 | L1 64 KiB(D)+64 KiB(I)/코어, L2 512 KiB/코어, L3 2 MiB 공유 | `/sys/devices/system/cpu/cpu0/cache/` |
| GPU | VideoCore VII (V3D 7.1) | [`paper_ko.md`](../10_paper/paper_ko.md) §3.1 |
| Vulkan 드라이버 | **Mesa V3DV 26.2.2**, apiVersion **1.3** | [`paper_ko.md`](../10_paper/paper_ko.md) §3.1 / 부록 A.1 |
| NPU | **Hailo-8** | [`paper_ko.md`](../10_paper/paper_ko.md) §3.1 |
| NPU 런타임 | **HailoRT 4.24.0** | [`paper_ko.md`](../10_paper/paper_ko.md) 부록 A.1 |
| PCIe 링크 | **8.0 GT/s x1 (Gen3)**, `0001:01:00.0` | [`display_e2e/env_bench.txt`](../07_results/display_e2e/env_bench.txt) |
| 비디오 디코더 | BCM2712 내장 **HEVC** 하드웨어 디코더 | [`paper_ko.md`](../10_paper/paper_ko.md) §3.1 |
| 디코드 라이브러리 | **libavcodec 61.19** (FFmpeg) | [`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) |
| ISP 라이브러리 | **libpisp 1.7.0** | [`isp_resize/summary.md`](../07_results/isp_resize/summary.md) 1부 |
| 커널 | **Linux 6.18.50+rpt-rpi-2712** | [`display_e2e/env_bench.txt`](../07_results/display_e2e/env_bench.txt) |
| 컴파일러 | **g++ 14.2**, `-O3 -mcpu=cortex-a76` | [`paper_ko.md`](../10_paper/paper_ko.md) 부록 A.1 |
| 디스플레이 | HDMI-A-1 **1920×1080@60**, labwc (Wayland) | [`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §1 |

> ⚠ **PCIe 세대와 클럭은 "지금도 그렇다"는 보장이 없다.** 이 표는 측정 당시의 스냅샷이므로
> 측정할 때는 반드시 sysfs에서 다시 읽는다
> ([`.claude/skills/measure/SKILL.md`](../.claude/skills/measure/SKILL.md) §1).
> **빈칸으로 둔 것**: DRAM 용량·규격, GPU 코어 클럭, V3D 연산 유닛 개수 — 프로젝트 문서에서
> 확인하지 못했다.

---

## 2.12 자가 점검

**Q1. Hailo-8(NPU)과 VideoCore VII(GPU)의 가장 중요한 차이는 "성능"인가?**

아니다. **위치**다. GPU는 BCM2712 SoC **안**에 있어서 DRAM을 바로 본다. Hailo-8은
**SoC 밖의 별개 칩**이고 **PCIe Gen3 x1**이라는 좁은 통로로 붙어 있다. 프레임 한 장당
약 2.4 MB(입력 1.23 MB + 출력 1.22 MB)를 그 통로로 왕복시켜야 하고, 이 왕복이 병목이다.
→ [`paper_ko.md`](../10_paper/paper_ko.md) §3.1·§3.3, [`npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md)

**Q2. PCIe 세대를 Gen3로 바꿨더니 FPS가 약 2배가 됐다. 이것만으로 "PCIe가 병목"이라 말할 수 있는가?**

부족하다. 결정적인 증거는 **시간을 분해했을 때**다. 한 배치의 처리 시간을
"배치당 고정비 + 프레임당 비용"으로 나누면 Gen2의 `약 46 + 32.6 ms`가
Gen3에서 `약 24.8 + 16.2 ms`로 **두 항이 모두 정확히 절반**이 됐다. 시간의 일부가 NPU 칩 안의
계산이었다면 그 부분은 링크와 무관하게 남아야 한다. 둘 다 반이 됐다는 것은
**시간의 거의 전부가 전송에 쓰인다**는 뜻이다. 다만 그것이 순수한 대역폭 포화인지
전송 고정 오버헤드인지는 **확정되지 않았다**(논문도 "전송 지배적임을 **시사한다**"라고만 쓴다).
→ [`npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md),
[`op_point/runs/latency_gen3_cooled/summary.md`](../07_results/op_point/runs/latency_gen3_cooled/summary.md) 결론 3

**Q3. "하드웨어 디코더를 쓰면 CPU가 안 든다"는 참인가?**

**거짓이다. 그 출력을 어떻게 받느냐가 정한다.** 같은 BCM2712 HEVC 하드웨어 디코더를
GStreamer `v4l2slh265dec`(rpt3)로 쓰면 **32.7 FPS / CPU 30.6 ms/frame**으로 소프트웨어 디코드
(`avdec_h265` 31.8 ms)와 거의 같다. CPU 시간의 약 90%가 프레임 복사·변환이었다
(원인은 문서가 "추정"으로 표시). 같은 하드웨어를 ffmpeg `-hwaccel drm`(CLI)으로 쓰면
**332.8 FPS / CPU 2.0 ms**이고, 이것도 대부분 CPU 내려받기다. 복사가 없으면 0.2~0.5 ms대다
(rpt4 GStreamer DMABuf 0.24, ffmpeg DRM_PRIME 유지 0.55, 앱 0.23~0.28). **차이는 디코더가 아니라 복사의 유무다.**
→ [`decode_benchmark/summary.md`](../07_results/decode_benchmark/summary.md) 해석 1, 회차 `runs/gst_rpt4`·`runs/ffmpeg_outfmt`

**Q4. CPU와 GPU가 "같은 메모리"를 쓰는데, 왜 GPU가 쓴 결과를 CPU로 읽으면 비싼가?**

같은 DRAM이어도 **매핑 방식**이 다르기 때문이다. V3D가 내놓는 Vulkan 메모리 타입에는
**`HOST_CACHED`가 없다** — CPU가 그 영역을 캐시된 상태로 읽을 방법이 드라이버에 없다.
write-combine 매핑은 **쓰기는 모아서 빠르지만 읽기는 매번 DRAM까지 간다**(방향 비대칭).
실측으로 순차 읽기 약 1.41 GB/s, stride-3 읽기 **약 0.088 GB/s**이고, 1.2 MB memcpy가
캐시 매핑 dma-heap 0.229 ms 대 V3D 메모리 1.045 ms로 **4.6배**다. 그래서 이 프로젝트는
"GPU가 쓴 버퍼를 CPU로 훑는 일"을 설계에서 없앴고, 그게 CPU 17.6 → 3.5 ms/frame의 본질이다.
→ [`paper_ko.md`](../10_paper/paper_ko.md) §3.3 제약 1, [`07_results/README.md`](../07_results/README.md) 2번,
[`isp_resize/summary.md`](../07_results/isp_resize/summary.md) 1부 표 2, 상세는 [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md)

**Q5. 측정 전에 칩이 식을 때까지 기다리는 것은 과한 신중함인가?**

아니다. Hailo-8은 연속 부하에서 **92.7 °C**까지 오르고 그러면 클럭이 **400 → 250 MHz**로 떨어진다.
이 상태에서 잰 지연은 냉각 상태보다 **4~7% 부풀려져 있었다**(b1 HW −6.1%, b8 Overall −7.0%).
게다가 이 프로젝트에는 그 탓에 "신뢰 불가"로 **버려진 측정 표가 실제로 있다.** 그래서 프로토콜이
"75 °C 초과면 65 °C까지 대기", "로그에 온도·클럭 경고가 1건이라도 있으면 신뢰 불가 표시"를 못박아 뒀다.
→ [`op_point/runs/latency_gen3_cooled/summary.md`](../07_results/op_point/runs/latency_gen3_cooled/summary.md) §3·§4,
[`npu_benchmark/runs/gen3/summary.md`](../07_results/npu_benchmark/runs/gen3/summary.md),
[`.claude/skills/measure/SKILL.md`](../.claude/skills/measure/SKILL.md) §2

---

**다음** → [03_yolo_tensor.md](03_yolo_tensor.md) (NPU가 뱉는 숫자 덩어리는 무엇인가),
NPU로 바로 가려면 [04_npu_hailort.md](04_npu_hailort.md).
관련: 메모리·캐시 상세는 [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md),
GPU 셰이더 상세는 [05_vulkan_compute.md](05_vulkan_compute.md), 용어는 [99_glossary.md](99_glossary.md).
