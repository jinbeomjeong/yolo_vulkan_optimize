# 04. NPU와 HailoRT — 추론을 대신 해 주는 칩과 그 사용법

> **선수 지식**: [01_big_picture.md](01_big_picture.md) (처리량·지연·병목·슬롯),
> [02_hardware.md](02_hardware.md) (칩들의 역할 분담),
> [03_yolo_tensor.md](03_yolo_tensor.md) (텐서, NHCW 레이아웃, 6개 출력 텐서)
> **다 읽으면 읽을 수 있는 것**: [`01_src/hailo/hailo_raw_runner.hpp`](../01_src/hailo/hailo_raw_runner.hpp)와
> [`hailo_runner.hpp`](../01_src/hailo/hailo_runner.hpp) 머리 주석,
> [`07_results/npu_benchmark/`](../07_results/npu_benchmark/),
> [`07_results/op_point/summary.md`](../07_results/op_point/summary.md),
> [`07_results/nms_host_cost/summary.md`](../07_results/nms_host_cost/summary.md)

---

## 4.0 이 챕터가 답하는 질문

[01_big_picture.md](01_big_picture.md) §1.2 (5)에서 이렇게 못 박았다.

> 이 프로젝트의 병목은 **NPU로 데이터를 실어 나르는 PCIe 통로**다.
> 그래서 이 프로젝트는 처음부터 **FPS를 올리는 것을 목표로 하지 않았다.**

이 챕터는 그 판단이 어떻게 **측정으로 확인됐는지**를 보인다. 다 읽으면
"왜 배치를 키워도 소용없는데 슬롯은 소용이 있는가", "왜 온칩이라고 믿었던 NMS가
사실 CPU에서 돌고 있었는가", "왜 크기를 잘못 주면 프로그램이 에러 없이 영원히 멈추는가"에
답할 수 있다.

---

## 4.1 NPU란 무엇인가

**NPU** = Neural Processing Unit(신경망 처리 장치). 신경망 연산만 빠르게 하려고 만든 칩이다.

### CPU / GPU / NPU 한 줄씩

| | 설계 목표 | 잘하는 일 | 못하는 일 |
|---|---|---|---|
| **CPU** | 무엇이든 빠르게 | 복잡한 조건 분기, 순차 논리, 한 가닥 작업의 속도 | 같은 계산 백만 번 |
| **GPU** | 같은 계산을 많은 데이터에 | 픽셀 백만 개에 같은 연산, 임의의 계산 | 가닥마다 다른 분기 |
| **NPU** | **행렬 곱만** 극한으로 | 합성곱·행렬 곱, 8비트 정수 연산 | **그 외 거의 전부** |

### 왜 신경망만 빠른가 — 세 가지 이유

**(1) 연산의 종류를 줄였다.** [03_yolo_tensor.md](03_yolo_tensor.md) §3.3에서 본 대로
CNN이 하는 일의 대부분은 **곱하고 더하기**(multiply-accumulate, MAC) 하나다. 범용 CPU는
수천 가지 명령을 지원해야 해서 명령 해독기, 분기 예측기, 캐시 계층에 칩 면적의 대부분을
쓴다. NPU는 그것을 거의 전부 버리고 **면적을 곱셈기로 채운다.**

**(2) 8비트 정수만 쓴다.** [03_yolo_tensor.md](03_yolo_tensor.md) §3.6에서 본 양자화 덕분이다.
8비트 정수 곱셈기는 32비트 부동소수점 곱셈기보다 회로가 수십 배 작다. 같은 면적에
곱셈기를 수십 배 더 넣을 수 있다.

**(3) 데이터를 옮기지 않는다.** 일반 프로세서는 "메모리에서 값을 읽고 → 계산하고 → 메모리에
쓰고"를 반복한다. 계산보다 **데이터 이동이 더 비싸다.** NPU는 연산기를 격자로 배열하고
**중간 결과를 옆 연산기로 직접 넘긴다.** 메모리를 거치지 않는다.

```
일반 프로세서                    NPU (데이터플로 구조)

 메모리 ─┐                       ┌─→[MAC]→[MAC]→[MAC]→┐
         ├→[ALU]→┐              │   ↓     ↓     ↓     │
 메모리 ←┘       │      메모리 ─┤  [MAC]→[MAC]→[MAC]  ├→ 메모리
         ←───────┘              │   ↓     ↓     ↓     │
                                 └─→[MAC]→[MAC]→[MAC]→┘
 매 연산마다 메모리 왕복          한 번 넣고 한 번 받는다
```

**대가**: NPU는 자기가 아는 연산만 한다. 모델에 NPU가 모르는 연산이 하나라도 있으면
그 부분은 CPU로 떨어진다. §4.6 (3)에서 보듯 **이 프로젝트에서 실제로 그런 일이 일어났다.**

---

## 4.2 컴파일 기반 실행 — HEF와 컨텍스트

### HEF란 무엇인가

**HEF** = Hailo Executable Format. 확장자 그대로 `03_models/yolov11m.hef` 파일이다.

일반적인 딥러닝 추론은 "모델 파일(가중치)을 읽고, 프레임워크가 그때그때 해석해서 실행"한다.
Hailo-8은 그렇게 하지 않는다.

> **HEF는 모델을 그 칩 전용 명령으로 미리 번역해 둔 것**이다.
> 가중치뿐 아니라 **"어느 연산기에 무엇을 싣고, 데이터를 어느 경로로 흘릴지"** 까지
> 전부 결정되어 들어 있다.

이 번역을 **컴파일(compile)** 이라고 하고, Hailo Dataflow Compiler가 한다.
중요한 제약이 둘 있다.

- **컴파일은 x86 호스트에서만 가능하다.** Raspberry Pi에서는 못 한다.
  즉 **HEF의 설정을 바꾸려면 이 보드를 떠나야 한다.**
- 그래서 "출력 형식을 바꿔 볼까?" 같은 실험을 가볍게 할 수 없다. 이 제약이
  §4.6 (4)의 논증을 **꼭 필요하게** 만들었다 — 재컴파일 없이 원하는 것을 얻을 수 있는지를
  먼저 증명해야 했다.

### 컨텍스트(context)가 4개라는 말의 뜻

`hailortcli parse-hef`로 이 HEF를 보면 **Multi Context, 컨텍스트 4개**라고 나온다
([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §3.2).

**컨텍스트 = 칩에 한 번에 올릴 수 있는 만큼의 모델 조각.**

YOLOv11m의 가중치 전부를 Hailo-8의 온칩 메모리에 올릴 수 없다. 그래서 컴파일러가 모델을
네 조각으로 자른다. 실행은 이렇게 진행된다.

```
[컨텍스트 1 가중치 적재] → 레이어 앞부분 실행 → 중간 결과 보관
        ↓ 가중치 교체 (PCIe로 새 가중치가 들어온다)
[컨텍스트 2 가중치 적재] → 다음 부분 실행
        ↓ 가중치 교체
[컨텍스트 3 가중치 적재] → ...
        ↓ 가중치 교체
[컨텍스트 4 가중치 적재] → 마지막 부분 → 출력 6텐서
```

**컨텍스트 전환(context switch)마다 가중치가 PCIe를 건너온다.** 이것이 이 프로젝트에서
PCIe가 병목이 된 가장 큰 이유다. 모델이 칩에 한 번에 다 들어갔다면(단일 컨텍스트)
가중치는 처음 한 번만 올리면 되고, PCIe로는 입력 이미지와 출력 텐서만 오갔을 것이다.

§4.6 (5)에서 보듯 측정된 **"배치당 고정비"의 정체가 바로 이 가중치 재적재**로 해석된다.

---

## 4.3 PCIe — 칩과 칩을 잇는 통로

**PCIe** = Peripheral Component Interconnect Express. 컴퓨터 안에서 칩과 칩을 잇는
고속 통로의 표준이다. 그래픽 카드, SSD, 그리고 이 Hailo-8 NPU가 여기에 붙는다.

> [02_hardware.md](02_hardware.md) §2.5가 PCIe 자체를 이미 다뤘다. 여기서는 **NPU를 쓸 때
> 무엇을 확인하고 무엇을 추측하면 안 되는지**만 짚고, §4.6 (6)에서 측정으로 돌아온다.

대역폭(초당 옮길 수 있는 바이트 수)을 정하는 값이 **두 개**다.

### (1) 세대(generation) — 전선 한 가닥의 속도

| 세대 | 신호 속도 | 한 레인당 실효 대역폭(대략) |
|---|---|---|
| Gen1 | 2.5 GT/s | 250 MB/s |
| **Gen2** | **5.0 GT/s** | 500 MB/s |
| **Gen3** | **8.0 GT/s** | ~985 MB/s |
| Gen4 | 16 GT/s | ~1.97 GB/s |

GT/s는 GigaTransfers per second(초당 기가 전송 횟수)다.

### (2) 레인 수(width) — 전선이 몇 가닥인가

`x1`, `x4`, `x16` 으로 적는다. **x4는 x1의 4배**를 동시에 보낸다.
Hailo-8 칩 자체는 Gen3 x4까지 지원하지만, **Raspberry Pi 5의 슬롯은 x1이다.**
즉 칩 성능의 4분의 1만 쓸 수 있다. 이것은 보드의 물리적 한계라 바꿀 수 없다.

### 현재 세대를 확인하는 법 — 추측하지 않는다

Pi 5는 기본적으로 Gen2로 동작하고, `/boot/firmware/config.txt`의 `dtparam=pciex1_gen=3`으로
Gen3으로 올릴 수 있다. **지금 어느 세대인지는 문서를 믿지 말고 읽어서 확인한다.**

```bash
cat /sys/bus/pci/devices/0001:01:00.0/current_link_speed   # 5.0 GT/s = Gen2, 8.0 GT/s = Gen3
cat /sys/bus/pci/devices/0001:01:00.0/current_link_width   # Pi 5 슬롯은 1
```

> **왜 이 규칙이 있는가**: 이 프로젝트는 과거에 문서에 적혀 있던 낡은 값을 그대로 인용해
> **Gen3 측정을 Gen2로 잘못 보고한 적이 있다.** 그래서 `07_results/` 아래 모든 측정
> summary가 첫 표에 "PCIe 링크: sysfs `current_link_speed`로 읽음"을 적는다
> ([`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §1 참고).
>
> `lspci -vv`의 `LnkSta:` 줄은 **쓰지 않는다.** root가 아니면 그 줄이 아예 출력되지 않아
> "확인했다"고 착각하기 쉽다.

### 이 프로젝트가 PCIe로 옮기는 양

| 방향 | 내용 | 바이트/프레임 |
|---|---|---:|
| 호스트 → NPU (H2D) | 입력 이미지 NHCW 640×640×3 | 1,228,800 |
| NPU → 호스트 (D2H) | 원시 출력 6텐서 | 1,221,120 |
| 그 밖 | **컨텍스트 4개의 가중치 재적재** | (프레임마다 반복) |

30 FPS면 입출력만으로도 초당 약 73 MB다. 여기에 가중치 트래픽이 더해진다.

---

## 4.4 배치(batch) — 여러 장을 묶어 보내기

**배치 크기 B** = 한 번의 추론 요청에 몇 장을 묶어 넣는가.

### 왜 묶으면 효율이 오르는가

§4.2에서 본 컨텍스트 전환 비용은 **"몇 장을 처리하든 한 번은 내야 하는 값"** 이다.
그렇다면 한 번 가중치를 올린 김에 여러 장을 처리하면 그 비용이 **나눠진다.**

```
배치 1 (8장을 따로)
 [가중치][1장][가중치][1장][가중치][1장] ... 8번
  ↑ 고정비를 8번 낸다

배치 8 (8장을 묶어)
 [가중치][1장][2장][3장][4장][5장][6장][7장][8장]
  ↑ 고정비를 1번만 낸다
```

### 무엇을 잃는가 — 지연

**8장이 다 모일 때까지 기다렸다가 한꺼번에 보내고, 8장의 결과가 다 나와야 받는다.**
첫 장 입장에서는 자기 혼자였으면 이미 끝났을 시간에 아직 대기 중이다.

[01_big_picture.md](01_big_picture.md) §1.2 (3)의 컨베이어 벨트 이야기와 정확히 같은
처리량–지연 맞바꿈이다. 이 프로젝트 실측이 그것을 그대로 보여 준다
([`07_results/npu_benchmark/gen3/summary.md`](../07_results/npu_benchmark/gen3/summary.md)).

| batch | FPS(처리량) | HW 지연 |
|---:|---:|---:|
| 1 | 24.34 | 32.51 ms |
| 8 | **51.71** | **126.33 ms** |

**처리량 2.1배를 얻는 대가로 지연이 3.9배 나빠졌다.**

그리고 배치를 계속 키운다고 계속 좋아지지 않는다. batch 16에서는 오히려 FPS가 떨어졌고
(43.72), `hailort.log`에 `Desc page size value (1024) is not optimal for performance` 경고가
찍혔다([같은 summary](../07_results/npu_benchmark/gen3/summary.md)).

> **미리 말해 두는 반전**: 위 표는 `hailortcli` 로 잰 **NPU 단독** 값이다.
> 실제 파이프라인에서는 **배치가 거의 무의미하고 슬롯이 지배한다.** §4.6 (6)에서 다룬다.

---

## 4.5 비동기 추론과 in-flight

### 동기 vs 비동기

**동기(synchronous)** 방식은 이렇다.

```
 [1장 보냄] ──── 결과 기다림(70 ms) ──── [받음] [2장 보냄] ──── 기다림 ──── [받음]
                 ↑ 이 동안 CPU와 GPU는 아무것도 안 한다. NPU도 뒷부분은 놀고 있다
```

**비동기(asynchronous)** 방식은 "보내고 나서 기다리지 않고 바로 다음 일을 한다".
결과는 나중에 **콜백(callback)** 으로 알려 준다.

```
 [1장 보냄][2장 보냄][3장 보냄][4장 보냄] ...
                     ↑ 그동안 GPU는 다음 프레임 전처리를 한다
      ↓ 나중에
 (콜백) 1장 완료   (콜백) 2장 완료   (콜백) 3장 완료
```

HailoRT에서 이 요청을 발행하는 함수가 **`infer_async()`** 다. 이 함수는 **요청을 걸어만 두고
즉시 반환한다.** 결과가 준비되면 HailoRT가 **자기 내부 스레드에서** 우리가 등록한 콜백 함수를
부른다. "내부 스레드에서 부른다"는 것이 §4.6 (9)의 규율로 이어진다.

### in-flight — 동시에 몇 장을 걸어 두는가

**in-flight**(비행 중) = **보냈지만 아직 결과가 안 온 프레임 수.**

이것이 [01_big_picture.md](01_big_picture.md)에서 말한 **슬롯(slot)** 의 실체다.
슬롯 하나가 "NPU에 걸어 둘 수 있는 자리 하나"에 대응한다.

```
 --slots 4 일 때

 슬롯 0: [프레임 12 전송 중]
 슬롯 1: [프레임 13 NPU 처리 중]     in-flight = 4
 슬롯 2: [프레임 14 NPU 처리 중]
 슬롯 3: [프레임 15 대기 중]
 → 슬롯이 다 찼다. 하나가 끝날 때까지 새로 못 보낸다 = 백프레셔(backpressure)
```

**백프레셔(backpressure, 역압)** 는 "받는 쪽이 감당 못 하면 보내는 쪽을 막는다"는
흐름 제어 장치다. 없으면 요청이 무한히 쌓여 메모리를 다 쓰거나 지연이 폭발한다.

### Little's law — 셋의 관계

큐 이론의 **Little's law**(리틀의 법칙)는 줄 서는 모든 시스템에 성립하는 아주 단순한 항등식이다.

> **in-flight ≈ 처리량 × 지연**

쉬운 말로 풀면 이렇다.

> **"안에 들어 있는 개수" = "1초에 몇 개 들어가는가" × "하나가 안에 머무는 시간"**

식당으로 비유하면: 손님이 **1분에 2명씩** 들어오고 **한 사람이 30분** 머무르면,
가게 안에는 늘 **약 60명**이 있다. `60 = 2 × 30`.

이 프로젝트는 이 식을 **측정값의 검산**에 쓴다.
[`07_results/raw_runner_backpressure/summary.md`](../07_results/raw_runner_backpressure/summary.md) §3의
`in-flight = FPS × npu_ms_mean / 1000` 열이 그것이다.

| 설정 | FPS | NPU 지연 | 계산된 in-flight | 설정한 slots |
|---|---:|---:|---:|---:|
| gpu b2/s4 | 34.96 | 110.31 ms | **3.86** | 4 |
| gpu b8/s16 | 51.48 | 297.16 ms | **15.29** | 16 |

**계산값이 슬롯 수에 거의 닿아 있다** → 백프레셔가 제대로 동작하고 NPU가 쉬지 않는다는 뜻이다.
이 검산이 실제로 버그를 잡아냈다. §4.6 (7)에서 다룬다.

---

## 4.6 이 프로젝트에서는

### (1) 두 개의 러너 — 고수준과 저수준

이 프로젝트에는 HailoRT를 쓰는 코드가 **두 벌** 있다. 하는 일이 다르다.

```
┌─ 기준 경로 (--postproc host) ──────────────────────────────────┐
│  01_src/hailo/hailo_runner.cpp        InferModel API (고수준)   │
│                                                                 │
│  VDevice → create_infer_model(hef) → set_batch_size             │
│         → configure() → Bindings::set_buffer                    │
│         → wait_for_async_ready() → run_async()                  │
│                                                                 │
│  출력: HEF 내장 NMS를 거친 검출 목록 1개 (FLOAT32, 160,320 B)   │
└─────────────────────────────────────────────────────────────────┘

┌─ 프로덕션 경로 (--postproc gpu) ───────────────────────────────┐
│  01_src/hailo/hailo_raw_runner.cpp    저수준 API                │
│                                                                 │
│  Hef::create → VDevice(ROUND_ROBIN)                             │
│         → create_configure_params(batch, 스트림 전부 ASYNC)     │
│         → VDevice::configure()                                  │
│         → 슬롯별 bind_slot(dma-buf fd 7개 매핑)                 │
│         → infer_async(입력 1 + 출력 6)                          │
│                                                                 │
│  출력: 원시 6텐서 (UINT8 NHCW, 합계 1,221,120 B) → GPU 후처리   │
└─────────────────────────────────────────────────────────────────┘
```

**`InferModel`이 더 쓰기 쉽고 권장되는 API다.** 그런데 이 프로젝트는 왜 저수준으로 내려갔나?

### (2) 원시 출력에는 `InferModel`을 쓸 수 없다

**확정된 사실이다.** `InferModel` 경로에서 출력 형식을 장치 원본(NHCW)으로 바꾸려고
`set_format_order(NHCW)`를 부르면, 그다음 `configure()`가 **`HAILO_INVALID_ARGUMENT`로
실패**하고 런타임이 `nms_post_process.cpp:42`를 원인으로 찍는다
([`07_results/gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md) 결론 1,
증거 파일 `evidence/infer_model_raw_attempt.txt`).

**이유는 이름 그대로다.** 이 HEF의 출력에는 `yolov8_nms_postprocess`라는 후처리 연산이
붙어 있고, `InferModel`은 그 연산을 **반드시 거치는 것을 전제로** 출력 형식을 결정한다.
"NMS를 건너뛴 원본을 내놔라"는 요청은 그 전제와 모순이라 거부된다.

그래서 저수준 경로(`VDevice::configure()` + `ConfiguredNetworkGroup::infer_async()`)로
내려가 **스트림을 직접 열었다.** 두 경로의 결과가 바이트 단위로 같다는 것은 별도로 확인했다
(`infer_async`와 `read_async` 두 방식이 40/40 프레임 바이트 일치,
[같은 summary](../07_results/gpu_postprocess/summary.md) P1 ③).

### (3) 스케줄러를 `ROUND_ROBIN`으로 켜야 한다 — 안 그러면 에러 72

VDevice를 만들 때 넘기는 파라미터 중 `scheduling_algorithm`의 **기본값이 `NONE`** 이다.
이 상태로 추론을 돌리면 **`HAILO_STREAM_NOT_ACTIVATED`(에러 코드 72)** 가 난다.

```cpp
// 01_src/hailo/hailo_raw_runner.cpp:36
vparams.scheduling_algorithm = HAILO_SCHEDULING_ALGORITHM_ROUND_ROBIN;
```

**스케줄러(scheduler)** 는 여러 네트워크·스트림이 한 장치를 나눠 쓸 때 순서를 정해 주는
장치다. `ROUND_ROBIN`(라운드 로빈)은 "차례대로 돌아가며"라는 뜻의 가장 단순한 방식이다.
스케줄러가 꺼져 있으면 스트림을 **직접 활성화**해 줘야 하는데, 비동기 API는 스케줄러가
켜져 있는 것을 전제로 만들어져 있다.

Python(`hailo_platform`)에서도 똑같다. `VDevice.create_params()`의 기본값이 NONE이므로
`params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN`을 반드시 설정한다
(프로젝트 [`CLAUDE.md`](../CLAUDE.md) 개발 환경 절).

### (4) HEF 내장 NMS는 **온칩이 아니라 호스트 CPU**에서 돈다

이 프로젝트에서 **가장 중요한 발견** 중 하나다.

**믿었던 것**: 출력이 `yolov8_nms_postprocess`니까 NPU가 칩 안에서 NMS까지 다 해서
깔끔한 검출 목록만 내보낸다.

**실제**: 그 연산은 **VStream 레벨의 호스트 연산**이다. 장치는 원시 텐서 1,221,120 B를
매 프레임 PCIe로 내보내고 있고, **호스트 CPU가 그것을 160,320 B로 줄인다.**

근거 4가지([`07_results/nms_host_cost/summary.md`](../07_results/nms_host_cost/summary.md) §1):

| 근거 | 내용 |
|---|---|
| HEF 사양 | `hailortcli parse-hef`가 원시 스트림 6개를 그대로 보여 준다 |
| gdb | 문제의 함수가 **HailoRT가 `std::thread`로 띄운 워커 스레드**에서 돈다. 진입 시 인자가 `0x50 = 80`으로 **클래스 개수와 일치** |
| 레이아웃 무관 | 입력을 NHWC로 주나 NHCW로 주나 3.60 vs 3.96 Mcycles/frame으로 거의 같다 → **출력 쪽 작업**이다 |
| 우리 코드 아님 | 우리가 짠 파싱·역변환 코드는 프로파일 0.1% 아래로 잡히지도 않는다 |

**그리고 이 비용은 검출 개수와 무관한 고정비다.** score threshold를 올려 검출을
2090개에서 0개로 없애도 CPU 시간이 그대로였다
([같은 summary](../07_results/nms_host_cost/summary.md) §2).

| score threshold | 검출 수 | CPU ms/frame |
|---|---:|---:|
| 0.2 | 2090 | 3.256 |
| 0.9 | 2 | 3.366 |
| 0.99 | **0** | **3.363** |

**왜 고정비인가**: 비용의 정체가 "겹친 박스를 억제하는 일"이 아니라
**"원시 텐서 전체를 훑는 일"** 이기 때문이다. [03_yolo_tensor.md](03_yolo_tensor.md) §3.8 (5)에서
본 그대로, 앵커 8400개 × (클래스 80 + DFL 64) ≈ **120만 개 값**을 매 프레임 역양자화하며
스캔해야 하고, **몇 개가 살아남든 이 일의 양은 같다.**

### (5) 그래서 원시 출력으로 바꿔도 PCIe 트래픽이 늘지 않는다

이 논증이 이 프로젝트의 방향을 정했다. §4.2에서 본 대로 **HEF 재컴파일은 x86에서만
가능**하므로, "원시 출력 HEF를 새로 만들어야 하나?"가 큰 갈림길이었다.

```
우리가 두려워한 시나리오                 실제

 장치 ──[160 KB]──> 호스트                장치 ──[1.22 MB]──> 호스트
        NMS는 칩 안에서 끝                        │
                                                   ▼ (같은 호스트 안에서)
 원시로 바꾸면 1.22 MB 가 되어             호스트 CPU가 NMS → 160 KB
 PCIe 트래픽이 7.6배!                      ↑ PCIe를 건넌 뒤의 일이다

 → 처리량이 폭락할 것이다                  → 우리가 NMS를 안 받겠다고 해도
                                              PCIe를 건너는 양은 **이미 1.22 MB**다
                                              → 트래픽은 **1바이트도 안 늘어난다**
```

**예측이 실측으로 확인됐다**([같은 summary](../07_results/nms_host_cost/summary.md) §4,
`hailortcli run2`로 두 모드를 교대 측정).

| 모드 | FPS 중앙값 | CPU ms/frame |
|---|---:|---:|
| `full_async` (호스트 NMS 포함) | 51.22 | 3.618 |
| **`raw_async` (원시 6스트림)** | **51.62** | **0.465** |

**처리량은 떨어지지 않았고(오히려 +0.78%), CPU가 프레임당 3.15 ms 줄었다.**

한편 더 이른 시점의 벤치마크도 같은 것을 이미 말하고 있었다.
`hw_only`(NPU만)와 `streaming`(호스트 처리 포함) FPS가 **0.2% 미만으로 같았다**
([`07_results/npu_benchmark/gen3/summary.md`](../07_results/npu_benchmark/gen3/summary.md)).
호스트 NMS가 아무리 비싸도 **처리량을 제한하지는 않고 있었다** — CPU 코어가 놀고 있었을 뿐이다.

> **그래서 얻은 것은 FPS가 아니라 CPU다.** 이 프로젝트 전체의 주제와 같다
> ([01_big_picture.md](01_big_picture.md) §1.6).

### (6) 큐 모델 `T(B) = 24.8 + 16.2·B` 와 PCIe 병목의 증거

배치 크기 B를 바꿔 가며 **한 배치를 처리하는 데 걸리는 시간**을 재고 직선으로 맞추면
이렇게 된다([`07_results/op_point/latency_gen3_cooled/summary.md`](../07_results/op_point/latency_gen3_cooled/summary.md) §4).

> **T(B) = 24.8 + 16.2 · B  [ms]**    (PCIe Gen3 x1, 냉각 상태)

두 항의 뜻:

- **24.8 ms = 배치당 고정비** — 배치 크기와 무관하게 한 번 내는 값.
  §4.2의 **컨텍스트 4개 가중치 재적재**로 해석한다.
- **16.2 ms = 프레임당 비용** — 장이 늘어날 때마다 붙는 값.
  입력 전송과 컨텍스트 경계의 중간 결과 왕복으로 해석한다.

**점근 상한**은 배치를 무한히 키웠을 때의 처리량이다. 고정비가 0으로 희석되므로

```
상한 FPS = 1 / 0.0162 s ≈ 61.6 FPS
```

실측 최고는 batch 8에서 **51.7 FPS**다
([`07_results/npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md)).
**이 프로젝트가 한 어떤 최적화도 이 천장을 올리지 못한다.**

#### Gen2 → Gen3 에서 두 항이 반씩 준 것이 왜 병목의 증거인가

같은 측정을 PCIe Gen2와 Gen3에서 각각 했다
([`07_results/npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md)).

| | 배치당 고정비 | 프레임당 비용 | 이론 상한 |
|---|---:|---:|---:|
| Gen2 x1 | ~46 ms | ~32.6 ms | ~30 FPS |
| Gen3 x1 | ~25 ms (정밀 적합 **24.8**) | ~16.2 ms | ~62 FPS |
| **비율** | **약 0.54배** | **0.50배** | |

**대역폭을 2배로 했더니 두 항이 모두 정확히 절반이 됐다.** 이것이 왜 결정적인가?

> 어떤 비용이 **연산**에 지배된다면, 통로를 넓혀도 그 비용은 변하지 않는다.
> 어떤 비용이 **전송**에 지배된다면, 통로를 2배로 넓히면 시간이 절반이 된다.
>
> **두 항이 모두 정확히 반으로 줄었다 = 두 항이 모두 전송 지배적이다.**

측정 전에는 "고정비는 가중치 전송이겠지만, 프레임당 비용은 실제 연산 시간 아닐까?"가
열린 질문이었다. 이 실험이 그것을 닫았다. **배치별 FPS도 전 구간에서 1.92~1.98배**로
거의 정확히 2배였다.

> **주의 — 두 모델을 섞지 마라.** `T(B) = 24.8 + 16.2·B`는 **`hailortcli`의 랜덤 입력
> 경로**를 설명하는 식이다. 실제 파이프라인에 적용하면 틀린다
> ([`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §4.3:
> 이 식으로 `b4/s4`를 예측하면 44.5인데 실측은 33.4다). 다음 항목이 그 이유다.

### (7) 처리량·지연을 지배하는 것은 batch가 아니라 **slots**다

§4.4에서 "배치를 키우면 처리량이 오른다"고 배웠다. **실제 파이프라인에서는 그렇지 않았다.**
정본은 [`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §4와 §8이다.

**같은 slots에서 batch를 바꿔도 거의 변화가 없다.**

| 설정 | FPS | NPU 지연 |
|---|---:|---:|
| `b4/s8` | 44.440 | 173.12 ms |
| `b8/s8` | 44.550 | 172.59 ms |
| | **차이 0.25%** | **사실상 동률** |
| `b2/s4` | **34.970** | 111.53 ms |
| `b4/s4` | 33.420 | 107.60 ms |
| | **작은 배치가 4.6% 빠르다** | |

**slots를 바꾸면 모든 것이 바뀐다.**

| slots | FPS | 프레임 주기 | NPU 지연 | **지연 ÷ 주기** |
|---:|---:|---:|---:|---:|
| 2 | 24.41 | 40.97 ms | 79.52 ms | **1.94** |
| 4 | 34.97 | 28.60 ms | 111.53 ms | **3.90** |
| 8 | 44.55 | 22.45 ms | 172.59 ms | **7.69** |
| 16 | 51.49 | 19.42 ms | 301.64 ms | **15.53** |

마지막 열을 보라. **지연 ÷ 주기 ≈ slots**가 네 지점 전부에서 성립한다. 이것을 뒤집으면
§4.5의 Little's law 그대로다.

```
 지연 ≈ slots × 프레임 주기
      = slots / 처리량                  (주기 = 1/처리량)
 ⟺  slots ≈ 처리량 × 지연  =  in-flight
```

**즉 `--slots`는 "큐 깊이"의 다른 이름이고, 그것이 곧 in-flight다.**
HailoRT는 걸려 있는 요청들을 자기가 알아서 묶어 처리하므로, 우리가 지정한 `batch`는
사실상 "스케줄러 버스트 크기" 힌트에 그친다.

> **실용적 결론**([같은 summary](../07_results/op_point/summary.md) §8.2):
> 배치의 천장에 닿은 뒤로 slots는 **순수하게 지연만 사는 손해 레버**다.
> `b2/s8`은 `b2/s4`와 FPS가 같은데(34.97) NPU 지연만 110 → 224 ms로 늘어난다.
> 그래서 배포 운용점이 `b2/s4`다.

### (8) 백프레셔 이중 계수 — in-flight 검산이 잡아낸 버그

§4.5의 Little's law 검산이 실제로 버그를 찾아낸 사례다.

GPU 후처리를 통합한 뒤 처리량이 눈에 띄게 떨어졌다. 원인은 후처리가 아니라
**`HailoRawRunner`의 백프레셔 이중 계수**였다.

```
 우리 코드:  increase_ongoing_callbacks()   ← +1  (수동)
             infer_async(...)               ← +1  (HailoRT 내부에서 이미 센다)
             ────────────────────────────────
             프레임당 +2

 → in-flight 상한이 num_slots 가 아니라 num_slots / 2 로 깎였다
 → 검산값이 slots 의 절반에서 멈춰 있었다  ← 이것이 증상이었다
```

수정은 **수동 증감 쌍을 제거**하는 것이었다
([`07_results/raw_runner_backpressure/summary.md`](../07_results/raw_runner_backpressure/summary.md) §2,
근거는 `evidence/ongoing_counter_sweep.txt`의 "프레임당 증가량 1 vs 2").
수정 후 in-flight가 슬롯 수에 닿았고(3.86/4, 15.29/16) 처리량이 host 경로 대비
**0.1% 이내**로 돌아왔다.

> ⛔ **수정 이전에 기록된 GPU 후처리 경로의 FPS·지연 값은 인용 금지다**
> ([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) 부록 A.4).
> 그 값들은 `07_results/gpu_postproc_e2e/`에 남아 있지만, 러너 버그가 섞인 수치라
> "GPU 후처리의 비용"을 전혀 대표하지 않는다. **후속 문서에서 쓰지 마라.**
> 데이터를 지우지 않고 "쓰지 말 것"으로 표시해 두는 것이 이 프로젝트의 관례다.

### (9) 크기 계약 — 틀리면 에러 없이 영원히 멈춘다

**이 챕터에서 가장 위험한 함정이다.**

dma-buf(여러 칩이 같은 메모리를 공유하는 수단, → [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md))를
HailoRT에 넘길 때 **크기를 두 군데에 적는데, 그 둘이 서로 다른 값이어야 한다.**

| 호출 | 넘겨야 하는 크기 | 왜 |
|---|---|---|
| `dma_map_dmabuf()` | **dma-buf 실크기** (`lseek(fd, 0, SEEK_END)`) | 매핑은 버퍼 전체를 덮는 쪽이 안전하다 |
| `infer_async()` | **`hw_frame_size`** | 실제로 전송할 바이트 수 |

두 값이 왜 다른가? GPU(V3DV)가 dma-buf를 할당할 때 내부 사정으로 **"+256 B 후 16 KB 단위
올림"** 을 하기 때문이다. 텐서 하나당 16~29 KB가 남는다
([`07_results/gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md) 주의 절).

**틀리면 어떻게 되는가:**

> `infer_async()`에 dma-buf 실크기(= 더 큰 값)를 주면, HailoRT는
> **`HAILO_SUCCESS`를 돌려준다.** 그런 다음 내부 워커 스레드가 그 요청을 **조용히 버린다.**
> **콜백이 단 한 번도 호출되지 않는다.**
>
> 완료를 무한정 기다리는 코드는 **영구 정지**한다. 로그에도 아무것도 안 남는다.

실제로 이 프로젝트의 첫 실행이 그렇게 멈췄고 **gdb로 붙어서야 알았다**
([같은 summary](../07_results/gpu_postprocess/summary.md) 주의 절).

**대책 두 가지**가 코드에 들어가 있다.

1. **모든 대기에 시한을 둔다.** [`hailo_raw_runner.cpp`](../01_src/hailo/hailo_raw_runner.cpp)의
   `drain_for(timeout)`이 그것이고, 시한을 넘기면 치명적 오류로 올린다.
   [`hailo_raw_runner.hpp`](../01_src/hailo/hailo_raw_runner.hpp) 머리 주석이
   **"그래서 `drain()`에 시한이 필수다"** 라고 못 박고 있다.
2. **매핑 크기가 `hw_frame_size`보다 작으면 시작 시점에 거부한다**
   (`hailo_raw_runner.cpp:195-205`).

> **일반화해서 기억할 것**: 성공을 반환하고 아무 일도 하지 않는 API가 실재한다.
> **"에러가 안 났으니 잘 됐겠지"는 검증이 아니다.** 결과가 실제로 도착했는지를
> 별도로 확인해야 하고, 확인할 수 없으면 최소한 시한을 걸어야 한다.

참고로 대체 경로인 `read_async()`는 같은 실수에 대해 **발행 시점에
`HAILO_STREAM_ABORT`(63)를 돌려준다.** 관측 가능한 실패가 조용한 실패보다 낫다는
좋은 대조 사례다.

### (10) `read_async`의 16384 B 정렬 요구

`read_async()`를 쓸 때 버퍼 주소가 **16384바이트(16 KB) 경계에 맞아야 한다.**
4096 B(4 KB)로 맞추면 동작은 하지만 프레임마다, 스트림마다
`vdma_stream.cpp:379`의 `unaligned buffer` 경고가 찍힌다. 첫 실행에서 **1,477건**이 나왔다
([같은 summary](../07_results/gpu_postprocess/summary.md) 주의 절).

**정렬(alignment)** 은 "데이터의 시작 주소가 어떤 수의 배수여야 한다"는 요구다.
DMA 엔진이 주소의 하위 비트를 무시하거나 페이지 단위로만 서술자를 만들 수 있기 때문에 생긴다.
정렬이 안 맞으면 런타임이 내부에서 복사를 하거나(느려짐) 경고를 낸다.

이 프로젝트는 `DmaBufExporter`의 `align` 파라미터로 처리했고, 최종 실행의
`unaligned buffer` 경고는 **0건**이다.

> 참고로 `hailort.log`는 HailoRT가 **실행한 디렉터리에** 남기는 로그다.
> 이 프로젝트의 측정 프로토콜은 매 실행의 `hailort.log`를 결과 폴더에 함께 저장하고
> **경고·에러 건수를 신뢰성 판정 기준으로 쓴다**
> ([`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §1 "신뢰성" 행:
> "18회 전부 온도·클럭 경고 0건, 에러 0건").

### (11) 콜백 스레드 규율 — 콜백 안에서 하면 안 되는 일

§4.5에서 본 대로 완료 콜백은 **HailoRT 내부 스레드**에서 호출된다. 우리 메인 스레드가
아니다. 여기에 규율이 셋 있다
([`hailo_raw_runner.hpp`](../01_src/hailo/hailo_raw_runner.hpp) "호출자 규약").

**① 콜백 안에서 Vulkan을 건드리지 마라.**
명령 제출, 펜스 대기, 큐 작업 전부 금지다. 이 프로젝트는 `VkQueue`가 **하나뿐**이라
다른 스레드에서 `vkQueueSubmit`을 하면 동기화가 깨진다. 그래서 NPU 완료분을 GPU로 미는
일까지 **메인 스레드**가 한다([`01_src/pipeline/main.cpp`](../01_src/pipeline/main.cpp)
머리 주석의 `--postproc gpu` 설명).

**② 콜백 안에서 다시 `submit()`하지 마라.**
제출 경로에는 백프레셔 대기(`wait_for_ongoing_callbacks_count_under`)가 들어 있다.
**콜백 스레드가 콜백을 기다리는** 꼴이 되어 교착(deadlock)할 수 있다.

**③ 짧게 유지하라.**
콜백은 "슬롯 번호를 free 큐에 넣고 나가는" 정도만 한다. 무거운 일을 하면 HailoRT의
내부 스레드가 묶여 다음 완료 처리가 밀린다.

한 가지 **좋은 소식**도 명시돼 있다.

> **콜백 진입 시점에 그 슬롯은 이미 반납된 상태다.** 버퍼 콜백 7개(입력 1 + 출력 6)가
> 모두 온 뒤에만 사용자 콜백이 불리므로, NPU는 그 슬롯의 dma-buf를 더 이상 만지지 않는다.
> 콜백 안에서 슬롯 번호를 다른 스레드로 넘겨 즉시 재사용해도 경합이 없다.

### (12) 입력은 NHCW로 넘긴다 — 여기서는 결론만

[03_yolo_tensor.md](03_yolo_tensor.md) §3.2에서 본 대로 장치 입력은 **NHCW**다.
응용이 NHWC로 주면 HailoRT가 **호스트 CPU에서 전치**해 준다.

이 프로젝트는 GPU 전처리 셰이더가 **처음부터 NHCW로 쓰게** 만들어 그 단계를 없앴다.
효과는 프레임당 CPU **17.6 ms → 3.5 ms**다
([`10_paper/paper_ko.md`](../10_paper/paper_ko.md) §4.3.2).

그런데 이 14 ms는 **전치 연산의 계산량으로 설명되지 않는다.** 같은 전치를 보통
메모리에서 하면 1.2 ms면 끝난다. 10배 이상의 차이가 어디서 오는지,
그리고 그것이 왜 이 프로젝트의 설계 원칙 전체를 정했는지는 다음 두 챕터가 다룬다.

- [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) — dma-buf와 GPU 메모리의 성질
- [10_why_this_design.md](10_why_this_design.md) — "GPU가 쓴 메모리를 CPU가 읽지 않는다"는 원칙

관련해서 `hailo_runner.hpp`에 적힌 제약 하나만 여기서 짚어 둔다.

> NHCW 입력은 **dma-buf에서만 쓸 수 있다.** HailoRT가 입력 버퍼를 직접 DMA 매핑하는데,
> V3D GEM mmap 포인터는 매핑에 실패한다
> (`Ioctl HAILO_VDMA_BUFFER_MAP failed due to invalid address`).
> ([`hailo_runner.hpp`](../01_src/hailo/hailo_runner.hpp) 머리 주석)

### (13) 발열 — 측정의 신뢰성을 무너뜨리는 변수

PCIe를 Gen3으로 올리고 나서 새 문제가 생겼다. **Hailo 칩이 최고 92.7 °C까지 올라갔고**,
`hailort.log`에 이 경고가 찍혔다
([`07_results/npu_benchmark/gen3/summary.md`](../07_results/npu_benchmark/gen3/summary.md)).

```
HEF was compiled assuming clock rate of 400 MHz, while the device clock rate is 250 MHz
```

칩이 스스로 클럭을 400 → 250 MHz로 낮춘 것이다. Gen2에서는 한 번도 나오지 않던 경고다.
SoC 쪽 `vcgencmd get_throttled`는 계속 0이었으므로 **Pi가 아니라 Hailo 칩 쪽의 클럭 저하**다.

과열 상태에서 잰 값은 지연을 **4~7% 부풀렸다**
([`07_results/op_point/latency_gen3_cooled/summary.md`](../07_results/op_point/latency_gen3_cooled/summary.md) §3).
그래서 이 프로젝트의 측정 프로토콜에는 **냉각 규칙**이 들어가 있다.

- 측정 전 칩 온도를 기준(65~68 °C) 이하로 내릴 때까지 대기
- 매 실행의 `hailort.log`에서 **온도·클럭 경고 건수를 세어 0이 아니면 그 측정을 버린다**
- 신뢰할 수 없는 옛 표는 지우지 않고 **"인용하지 마라"로 표시**해 둔다

> 이것이 [01_big_picture.md](01_big_picture.md) §1.5 (3)에서 말한 "시간은 전부 중앙값이다"와
> 같은 뿌리의 규율이다. 자세한 프로토콜은 [08_method.md](08_method.md)가 다룬다.

---

## 4.7 정리 — NPU 한 장이 흘러가는 길

```
   메인 스레드                       HailoRT                     Hailo-8 칩
   ───────────                       ───────                     ──────────
 빈 슬롯 확보
   │ (없으면 대기 = 백프레셔)
   ▼
 GPU 전처리 완료 대기
   │  NHCW 640×640×3 dma-buf (1,228,800 B)
   ▼
 infer_async(입력 1 + 출력 6) ──────► 요청 큐에 적재
   │  ⚠ 크기는 hw_frame_size            │
   │  ⚠ 매핑은 dma-buf 실크기           │ 스케줄러(ROUND_ROBIN)가 순서 결정
   │                                     ▼
   │ 즉시 반환 → 다음 프레임 처리   ┌── PCIe H2D 전송 ─────────► 입력 적재
   │                                 │                            [컨텍스트 1] 가중치↓
   │  ... 이 동안 in-flight = slots  │                            [컨텍스트 2] 가중치↓
   │      만큼 프레임이 안에 있다   │                            [컨텍스트 3] 가중치↓
   │                                 │                            [컨텍스트 4] 가중치↓
   │                                 └── PCIe D2H 전송 ◄────────  원시 6텐서
   │                                     │                        (1,221,120 B)
   │                                     ▼
   │                              버퍼 콜백 7개 도착
   │                                     │ 전부 모이면
   │            ◄─── 사용자 콜백 ────────┘  (HailoRT 내부 스레드!)
   │                 · 슬롯을 free 큐에 넣는다
   │                 · Vulkan 금지, 재제출 금지, 짧게
   ▼
 (메인 스레드가) GPU 후처리 submit → yolo_decode.comp → CPU NMS
```

**병목은 오른쪽 세로줄 전체**다. 왼쪽에서 무엇을 개선해도 처리량은 오르지 않는다.
그래서 이 프로젝트는 왼쪽에서 **CPU 시간**을 얻는 데 집중했다.

---

## 4.8 자가 점검

**Q1. NPU가 CNN을 CPU보다 빠르게 돌리는 이유를 세 가지로 말하라. 대가는 무엇인가?**

① 연산 종류를 MAC(곱하고 더하기) 하나로 줄여 칩 면적을 전부 곱셈기로 채웠다,
② 8비트 정수만 써서 같은 면적에 연산기를 수십 배 넣었다,
③ 중간 결과를 메모리에 쓰지 않고 옆 연산기로 직접 넘긴다.
대가는 **자기가 아는 연산만 한다**는 것이고, 실제로 이 모델의 NMS는 NPU가 아니라
호스트 CPU가 하고 있었다(§4.6 (4)).

**Q2. 이 HEF의 "컨텍스트 4개"가 성능에 미치는 영향은?**

모델이 칩 메모리에 한 번에 안 들어가서 네 조각으로 잘렸다는 뜻이고,
**조각을 바꿀 때마다 가중치가 PCIe를 건너온다.** 측정된 큐 모델
`T(B) = 24.8 + 16.2·B`의 **배치당 고정비 24.8 ms**가 이 재적재로 해석된다
([`07_results/op_point/latency_gen3_cooled/summary.md`](../07_results/op_point/latency_gen3_cooled/summary.md) §4).

**Q3. PCIe를 Gen2에서 Gen3으로 바꿨을 때 고정비와 프레임당 비용이 둘 다 절반이 됐다.
이것이 "PCIe가 병목"의 증거가 되는 이유는?**

연산에 지배되는 비용은 통로를 넓혀도 안 변하고, **전송에 지배되는 비용만 통로 폭에
반비례한다.** 두 항이 모두 정확히 반으로 줄었으므로(46→25, 32.6→16.2) 두 항이 모두
전송 지배적이다. 배치별 FPS도 전 구간 1.92~1.98배였다
([`07_results/npu_benchmark/summary.md`](../07_results/npu_benchmark/summary.md)).
현재 링크 세대는 문서가 아니라 `/sys/bus/pci/devices/0001:01:00.0/current_link_speed`로 읽는다.

**Q4. 파이프라인에서 `--batch`를 2에서 8로 올렸는데 FPS가 거의 안 변했다. 왜인가?
무엇을 올려야 하는가?**

처리량을 지배하는 것은 batch가 아니라 **slots**(= 큐 깊이 = in-flight)이기 때문이다.
같은 slots에서 batch 차이는 0.25%뿐이다(`b4/s8` 44.440 vs `b8/s8` 44.550).
slots를 올리면 FPS가 오르지만 **지연 ≈ slots × 프레임 주기**로 같이 나빠진다
([`07_results/op_point/summary.md`](../07_results/op_point/summary.md) §4, §8.2).
다만 배치 천장에 닿은 뒤의 slots는 지연만 사는 손해 레버다.

**Q5. Little's law를 이 파이프라인의 말로 쓰고, 그것이 어떤 버그를 잡았는지 말하라.**

**in-flight ≈ FPS × NPU 지연.** 즉 "NPU 안에 들어 있는 프레임 수 = 초당 넣는 수 × 한 장이
머무는 시간"이다. 이 검산값이 설정한 slots의 **절반**에서 멈춰 있는 것이 이상해서 파고들었더니,
러너가 백프레셔 카운터를 **이중으로 세고 있었다**(수동 증감 + `infer_async` 내부 계수).
수동 쌍을 제거하니 in-flight가 slots에 닿았다
([`07_results/raw_runner_backpressure/summary.md`](../07_results/raw_runner_backpressure/summary.md)).

**Q6. `infer_async()`에 dma-buf 실크기를 넘기면 어떤 일이 벌어지는가? 어떻게 방어하는가?**

**`HAILO_SUCCESS`를 반환하고 콜백을 0회 호출한다.** 요청이 내부 워커에서 조용히 버려져서,
완료를 무한정 기다리는 코드는 **영구 정지**한다. 에러도 로그도 없다.
전송 크기는 반드시 **`hw_frame_size`**, 매핑 크기는 **dma-buf 실크기**다.
방어는 ① 모든 대기에 시한을 두고 시한 초과를 치명적 오류로 올리기(`drain_for`),
② 매핑 크기가 `hw_frame_size`보다 작으면 시작 시점에 거부하기다
([`hailo_raw_runner.hpp`](../01_src/hailo/hailo_raw_runner.hpp) "크기 계약").

**Q7. 원시 출력(1.22 MB)이 NMS 출력(160 KB)보다 7.6배 큰데도 PCIe 트래픽이 늘지 않는 이유는?**

`yolov8_nms_postprocess`가 **호스트 CPU에서 도는 연산**이기 때문이다. 장치는 어느 쪽이든
원시 텐서 1,221,120 B를 PCIe로 내보내고 있고, 줄이는 일은 그것이 건너온 **뒤에** 호스트에서
일어난다. 우리가 NMS를 안 받겠다고 해도 건너오는 양은 그대로다.
실측으로 `raw_async` 51.62 FPS vs `full_async` 51.22 FPS로 오히려 미세하게 빨랐고
CPU는 3.618 → 0.465 ms/frame으로 줄었다
([`07_results/nms_host_cost/summary.md`](../07_results/nms_host_cost/summary.md) §3, §4).

**Q8. 완료 콜백 안에서 GPU 명령을 제출하면 안 되는 이유는?**

콜백은 **HailoRT 내부 스레드**에서 불리는데, 이 프로젝트는 `VkQueue`가 하나뿐이라
다른 스레드에서 제출하면 동기화가 깨진다. 같은 이유로 콜백 안에서 재제출도 금지다 —
제출 경로의 백프레셔 대기가 **콜백 스레드가 콜백을 기다리는** 교착을 만든다.
콜백은 슬롯을 free 큐에 넣고 즉시 나가야 한다
([`hailo_raw_runner.hpp`](../01_src/hailo/hailo_raw_runner.hpp) "호출자 규약").

---

**다음** → [05_vulkan_compute.md](05_vulkan_compute.md) (이 텐서를 만들고 해석하는 GPU 쪽)
또는 두 칩이 어떻게 같은 메모리를 공유하는지 → [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md)
