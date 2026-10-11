# 05. Vulkan 컴퓨트 — 그래픽스 카드에게 계산을 시키는 법

> **선수 지식**: [01_big_picture.md](01_big_picture.md) (처리량·지연·병목·슬롯·zero-copy 용어),
> [02_hardware.md](02_hardware.md) (칩 다섯 개가 각각 무엇을 잘하는가)
> **다 읽으면 읽을 수 있는 것**: [`02_shaders/sand_letterbox.comp`](../02_shaders/sand_letterbox.comp)
> 와 [`02_shaders/yolo_decode.comp`](../02_shaders/yolo_decode.comp) 의 머리 주석,
> [`01_src/vulkan/vulkan_context.hpp`](../01_src/vulkan/vulkan_context.hpp),
> [`cmake/YvoShaders.cmake`](../cmake/YvoShaders.cmake) 의 `yvo_add_shaders()` 함수

---

## 5.1 왜 하필 그래픽스 API로 계산을 하는가

### CPU는 똑똑하고 GPU는 많다

CPU(Central Processing Unit, 중앙처리장치)는 **한 번에 한 가지 일을 아주 잘한다.**
"이 값이 5보다 크면 이쪽으로 가고 아니면 저쪽으로 가라" 같은 복잡한 판단을 빠르게 한다.
이 보드의 CPU는 Cortex-A76 코어가 4개다. 즉 동시에 4가지 일을 할 수 있다(스레드를 더 만들어도
물리적으로 돌아가는 것은 4개다).

GPU(Graphics Processing Unit, 그래픽 처리 장치)는 반대다. **같은 계산을 수십만 개 데이터에
동시에 하는 데 특화**돼 있다. 대신 각각은 CPU보다 느리고, 조건 분기(if-else)를 싫어한다.

왜 그렇게 만들어졌나? 원래 목적이 **화면 그리기**였기 때문이다. 1920×1080 화면은 픽셀이
207만 개다. 각 픽셀에 "이 픽셀의 색은 무엇인가"를 계산해야 하는데, **207만 개 전부 같은 계산**이다.
CPU로 하면 207만 번을 순서대로 도는데 GPU는 수백 개를 동시에 돈다.

우리 문제가 정확히 이 모양이다.

- 전처리: 1920×1080 YUV 픽셀 하나하나를 RGB로 바꾸고 축소한다 → **같은 계산 × 207만**
- 후처리: 격자칸 8400개마다 "여기에 뭔가 있나"를 판정한다 → **같은 계산 × 8400**
- 화면 변환: 207만 픽셀을 모니터 형식으로 바꾼다 → **같은 계산 × 207만**

전부 GPU 모양이다.

### "셰이더"라는 이상한 이름

**셰이더(shader)** 는 GPU에서 도는 작은 프로그램이다. 이름이 이상한 이유는 역사다 —
처음에는 진짜로 "그림자(shade)와 음영을 계산하는 프로그램"이었다. 3D 게임에서 벽에 빛이
어떻게 떨어지는지를 픽셀마다 계산하던 것이다.

시간이 지나면서 사람들이 눈치챘다. "이거 그림 안 그리고 그냥 숫자 계산만 시키면 안 되나?"
그래서 만들어진 것이 **컴퓨트 셰이더(compute shader)** 다. 화면도 삼각형도 텍스처도 없이,
**버퍼(숫자 배열)를 읽어서 버퍼에 쓰는 것만 하는 셰이더**다.

이 프로젝트의 셰이더 5개는 전부 컴퓨트 셰이더다. 그림을 "그리는" 셰이더는 하나도 없다
(박스를 그리는 셰이더조차 컴퓨트다 — 이유는 §5.10에서 설명한다).
소스는 5개지만 SPIR-V 산출물은 6개다 — `overlay_boxes.comp` 를 storage-buffer 변형으로 한 번 더 컴파일한다(이 장 끝 빌드 파이프라인).

### CUDA도 OpenCL도 아니고 왜 Vulkan인가

GPU 계산을 시키는 방법은 여러 가지다.

| 방법 | 만든 곳 | 이 보드에서 |
|---|---|---|
| **CUDA** | NVIDIA | **불가능.** NVIDIA GPU 전용이다. 이 보드 GPU는 Broadcom VideoCore VII다 |
| **OpenCL** | Khronos (개방 표준) | 이 보드의 V3D 드라이버(Mesa)에 쓸 만한 구현이 없다 |
| **Vulkan 컴퓨트** | Khronos (개방 표준) | **된다.** Mesa의 V3DV 드라이버가 Vulkan 1.3을 지원한다 |
| **OpenGL ES 컴퓨트** | Khronos | 되지만 dma-buf 공유·세밀한 동기화 제어가 Vulkan만 못하다 |

즉 **선택의 여지가 별로 없었다.** "Vulkan이 제일 좋아서"가 아니라 **"이 보드에서 GPU를
제대로 쓸 수 있는 길이 Vulkan뿐"** 이라서 Vulkan이다.

그리고 결과적으로 이것이 행운이었다. 이 프로젝트의 핵심은 §5.11과
[06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) 에서 다룰 **메모리 공유(zero-copy)** 인데,
그 기능(external memory, dma-buf import/export)을 가진 API가 Vulkan이다.

---

## 5.2 Vulkan 객체 모델 — 무엇부터 만들어야 하는가

Vulkan은 "친절한" API가 아니다. GPU에 계산 하나를 시키려면 객체를 10개 넘게 만들어야 한다.
대신 **무슨 일이 언제 일어나는지가 전부 명시적**이다. 성능을 정밀하게 다뤄야 하는 이 프로젝트에는
이 성질이 필요하다.

만드는 순서는 정해져 있다. 각 단계가 앞 단계의 결과물을 요구하기 때문이다.

```
  VkInstance          "이 프로세스에서 Vulkan을 쓰겠다"
      │                (여기서 검증 레이어도 켠다)
      ▼
  VkPhysicalDevice    "이 컴퓨터에 GPU가 뭐가 있나" — 목록을 받아서 하나를 고른다
      │                ⚠ 이 보드에서는 목록에 2개가 나온다 (아래 참고)
      ▼
  VkDevice            "고른 GPU를 실제로 열겠다" — 쓸 기능·확장을 여기서 선언한다
      │                (논리 장치. 이 핸들로 거의 모든 것을 만든다)
      ▼
  VkQueue             "일을 제출할 창구" — VkDevice를 만들 때 함께 받는다
```

### ⚠ 물리 장치 목록에 GPU가 아닌 것이 섞인다

이 보드에서 물리 장치 목록을 받으면 **두 개**가 나온다.

1. **V3D** — 진짜 GPU(VideoCore VII)
2. **llvmpipe** — GPU가 아니라 **CPU로 Vulkan을 흉내 내는 소프트웨어 구현**이다

llvmpipe를 고르면 코드는 멀쩡히 돌아가지만 **실제로는 CPU가 셰이더를 해석해서 실행**한다.
성능을 재고 있었다면 측정값 전체가 무의미해진다. 그래서 이 프로젝트는 장치 이름에 `"V3D"` 가
들어간 것만 고른다 ([`vulkan_context.hpp`](../01_src/vulkan/vulkan_context.hpp) 머리 주석).

### 커맨드 버퍼 — GPU에게 일을 시키는 방식

여기서 Vulkan이 낯선 지점이 나온다. **GPU에게 "지금 이거 해" 라고 직접 말하는 함수가 없다.**

대신 이렇게 한다.

```
   1) VkCommandBuffer 를 연다                       vkBeginCommandBuffer
   2) 할 일을 "녹음"한다 (아직 실행 안 됨)
        vkCmdBindPipeline      — 어떤 셰이더를 쓸 것인가
        vkCmdBindDescriptorSets— 어떤 버퍼를 볼 것인가
        vkCmdPushConstants     — 작은 인자들
        vkCmdPipelineBarrier2  — 순서 규칙 (§5.7)
        vkCmdDispatch          — 실행해라
   3) 녹음을 닫는다                                  vkEndCommandBuffer
   4) 큐에 제출한다 → ★ 여기서 처음으로 GPU가 움직인다   vkQueueSubmit
   5) 끝났는지 확인한다                               vkWaitForFences
```

왜 이렇게 번거롭게 하나? **재사용** 때문이다. 매 프레임 같은 일을 한다면 커맨드 버퍼를 한 번만
녹음해 두고 계속 다시 제출하면 된다. 그리고 여러 단계를 **한 커맨드 버퍼에 몰아 녹음하면
제출이 한 번**이라 호출 비용이 줄어든다.

> 이 프로젝트의 표시 경로가 정확히 그렇게 한다. `display_convert` 와 `overlay_boxes` 가 **같은
> 커맨드 버퍼**에 이어 붙어 있고, 그래도 변환 시간이 느려지지 않는다는 것을 실측으로 확인했다
> (4.229 → 4.234 ms, 검출 5개 기준, V3D 960 MHz 조건. [`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 8).

### ★ VkQueue는 "외부 동기화 객체"다 — 이 프로젝트 전체 구조의 근거

Vulkan 스펙은 각 함수의 인자에 **"externally synchronized"(외부 동기화)** 라는 표시를 붙인다.
뜻은 이렇다 — **"이 객체를 두 스레드가 동시에 만지면 안 된다. Vulkan이 락을 걸어 주지 않으니
너희가 알아서 막아라."**

`vkQueueSubmit` 의 `VkQueue queue` 인자가 바로 그 표시가 붙은 인자다
(vk.xml, Vulkan 1.4.309: `<param externsync="true">VkQueue queue</param>`).
`vkQueueSubmit2`, `vkQueuePresentKHR` 도 마찬가지다.

그런데 이 프로젝트에는 **큐가 하나뿐이다**(`VulkanContext::queue()`).
그리고 GPU에 제출하는 곳은 여러 군데다 — 전처리, 후처리, 표시 변환, 오버레이.

여기서 규율이 나온다.

> **Vulkan을 만지는 일은 전부 메인 스레드에서만 한다. 새 스레드를 만들지 않는다.**
> — [`01_src/pipeline/infer_stage.hpp`](../01_src/pipeline/infer_stage.hpp) 머리 주석
> "스레드 규율 (이것이 전체 설계를 정한다)"

이 한 줄 때문에 파이프라인 구조가 통째로 정해졌다. NPU 완료 알림은 HailoRT의 콜백 스레드에서
오는데, 그 콜백에서 바로 GPU 후처리를 제출할 수 없다. 그래서 콜백은 **"끝났다"는 사실만 큐에
넣고**, 메인 스레드가 나중에 `pump()` 를 돌면서 그것을 GPU로 민다. 같은 이유로
[`main.cpp`](../01_src/pipeline/main.cpp) 머리 주석에도 같은 문장이 있다:
"VkQueue가 하나뿐이라 vkQueueSubmit을 다른 스레드에서 하면 안 되기 때문이다."

---

## 5.3 셰이더 실행 모델 — 워크그룹, invocation, subgroup

컴퓨트 셰이더를 실행하라고 시킬 때 쓰는 명령이 **디스패치(dispatch)** 다.

```c
vkCmdDispatch(cb, 40, 160, 1);   // 워크그룹을 x방향 40개, y방향 160개 띄워라
```

이때 실제로 GPU에서 벌어지는 일의 계층은 세 단계다.

```
   ┌─ 디스패치 (한 번의 vkCmdDispatch) ─────────────────────────────┐
   │                                                                │
   │  ┌ 워크그룹 (workgroup) ┐  ┌ 워크그룹 ┐  ┌ 워크그룹 ┐  ...      │
   │  │  ┌──┬──┬──┬──┐       │  │          │  │          │          │
   │  │  │in│in│in│in│ ← invocation (셰이더 코드가 한 번 도는 것)     │
   │  │  ├──┼──┼──┼──┤       │  │          │  │          │          │
   │  │  │in│in│in│in│       │  │          │  │          │          │
   │  │  └──┴──┴──┴──┘       │  │          │  │          │          │
   │  │  └─ subgroup(16개) ─┘ │  │          │  │          │          │
   │  └──────────────────────┘  └──────────┘  └──────────┘          │
   └────────────────────────────────────────────────────────────────┘
```

- **invocation(인보케이션)**: 셰이더 `main()` 이 **한 번 도는 것**. 가장 작은 단위다.
  픽셀 하나, 격자칸 하나가 보통 invocation 하나에 대응한다.
- **워크그룹(workgroup)**: invocation을 묶은 덩어리. 크기는 셰이더 안에 `local_size_x/y/z` 로
  적는다. **같은 워크그룹 안의 invocation끼리는 공유 메모리(shared memory)를 쓰고
  `barrier()` 로 서로 기다릴 수 있다.** 다른 워크그룹끼리는 못 한다.
- **subgroup(서브그룹)**: 하드웨어가 실제로 **완전히 같은 박자로** 실행하는 lane 묶음.
  V3D에서는 16개다. 같은 subgroup 안의 lane들이 서로 다른 `if` 가지로 갈라지면
  하드웨어가 양쪽을 다 실행하고 한쪽 결과를 버린다 → **분기가 비싸다**는 말의 정체가 이것이다.

### 나는 몇 번째 invocation인가 — `gl_GlobalInvocationID`

셰이더 코드는 수만 개 invocation이 **똑같은 코드**를 돈다. 그러면 "나는 어느 픽셀을
맡았는가"를 어떻게 아는가? GLSL이 내장 변수로 알려준다.

| 변수 | 뜻 |
|---|---|
| `gl_LocalInvocationID` | 내 워크그룹 **안에서** 내 좌표 (0 .. local_size-1) |
| `gl_WorkGroupID` | 내 워크그룹의 번호 |
| **`gl_GlobalInvocationID`** | **디스패치 전체에서 내 좌표.** = WorkGroupID × local_size + LocalInvocationID |

실제 코드를 보자 ([`yolo_decode.comp`](../02_shaders/yolo_decode.comp)):

```glsl
const int wi = int(gl_GlobalInvocationID.x);   // 내가 맡은 워드 열
const int h  = int(gl_GlobalInvocationID.y);   // 내가 맡은 행
if (wi >= pc.w_words || h >= pc.grid_h) return;   // ★ 경계 검사
```

### 경계 검사는 왜 항상 필요한가

디스패치는 **워크그룹 단위**로만 띄울 수 있다. 데이터가 2100개인데 워크그룹 크기가 60이면
35개 워크그룹으로 딱 맞지만, 데이터가 2105개면 36개를 띄워야 하고 **마지막 워크그룹의
invocation 55개는 담당할 데이터가 없다.** 그대로 두면 배열 밖을 건드린다.

그래서 모든 셰이더 맨 앞에 `if (내 번호 >= 개수) return;` 이 있다.

**단, 함정이 하나 있다.** 워크그룹 안에서 `barrier()`(모두 여기까지 오면 같이 가자)를 쓴다면,
일부 invocation이 `barrier()` 전에 `return` 해 버리면 나머지가 **영원히 기다린다.**
앞의 셰이더 4개(전처리·디코드·표시 변환·오버레이)는 `barrier()` 와 shared memory를 **쓰지 않으므로**
조기 return이 안전하다. 그래서 셰이더 주석마다 이렇게 적혀 있다:

```glsl
if (gx >= groups_per_row || dy >= pc.dst_h) return; // barrier 없음
```

**예외는 [`efficient_nms.comp`](../02_shaders/efficient_nms.comp) 하나다.** 워크그룹 하나(256 invocation)가
프레임 한 장의 후보를 shared memory에 올려 정렬하고 억제하므로 `barrier()` 가 수십 번 나온다.
그래서 이 셰이더는 "내 몫이 없으면 return" 대신 **`continue` 로 건너뛰고 barrier에는 모두 도착한다.**
return은 전원이 같은 shared 값을 보고 함께 빠지는 곳(용량 초과 등)에만 있다 — 이것을
**균일 제어 흐름**(uniform control flow)이라 한다. 설계는 [11장](11_tensorrt_efficient_nms.md) §5.4 의 번역이다.

---

## 5.4 데이터를 셰이더에 넘기는 세 가지 수단

셰이더는 프로그램이니까 **입력**이 필요하다. Vulkan에는 성격이 다른 세 가지 경로가 있고,
셋을 구분해 쓰는 것이 성능에 직접 영향을 준다.

```
        크기          변경 시점            비용
  ┌──────────────┬────────────────┬──────────────────────────┐
  SSBO         MB~GB         매 프레임         버퍼 하나 = 디스크립터 하나
  push constant  ≤128 B      매 디스패치       커맨드 버퍼에 값이 박힌다(매우 쌈)
  spec constant  스칼라      파이프라인 생성 시  컴파일 타임 상수가 된다(공짜)
  └──────────────┴────────────────┴──────────────────────────┘
```

### (1) SSBO — 큰 데이터

**SSBO**(Shader Storage Buffer Object, 셰이더 스토리지 버퍼)는 셰이더가 읽고 쓸 수 있는
**큰 메모리 덩어리**다. 영상 프레임, 텐서, 검출 결과가 전부 SSBO로 오간다.

```glsl
layout(std430, set = 0, binding = 0) readonly buffer YBuf { uint y_words[]; };
```

`std430` 은 **메모리 배치 규칙(layout)** 이다. C++ 구조체와 GLSL 구조체의 바이트 위치가
정확히 같아야 하는데, GLSL은 기본적으로 제멋대로 패딩을 넣는다. `std430` 은 "C와 거의 같게
깔아라"는 규칙이고, SSBO에는 이것을 쓴다(UBO에는 `std140` 이라는 더 헐거운 규칙을 쓴다).

배치가 어긋나면 **에러 없이 틀린 값을 읽는다.** 그래서 이 프로젝트는 C++ 쪽에
`static_assert` 로 크기를 못 박아 둔다:

```cpp
static_assert(sizeof(SandLetterboxPush) == 56, "push constant layout must match shader");
```

> 참고: `uint y_words[]` 처럼 **바이트가 아니라 32비트 워드 배열로 읽는** 것은 V3D에 맞춘
> 의도적 선택이다. V3D에서 바이트 하나를 읽는 명령은 워드 하나를 읽는 것보다 **명령 수가 4배**다
> ([`sand_letterbox.comp`](../02_shaders/sand_letterbox.comp) 주석). 그래서 워드로 읽고
> 셰이더 안에서 시프트로 바이트를 꺼낸다.

### (2) push constant — 작고 자주 바뀌는 값

`push constant` 는 커맨드 버퍼에 **값이 그대로 박히는** 아주 작은 인자다. 버퍼를 만들 필요도,
디스크립터를 갱신할 필요도 없다. 대신 **크기 한도가 작다**(스펙 한도는 `maxPushConstantsSize`,
이 프로젝트는 128 B 이내를 규율로 삼는다).

전처리 셰이더의 push constant는 56 B다 — 영상 크기, letterbox 여백, SAND 열 간격 같은
"프레임마다 달라질 수 있지만 작은" 값들이다.

### (3) specialization constant — 재컴파일 없이 코드를 바꾸는 상수

이게 가장 낯선 개념이다.

**specialization constant(스페셜라이제이션 상수, 이하 spec constant)** 는 GLSL 소스에는
"변수"처럼 적혀 있지만, **파이프라인을 만드는 시점에 값이 정해지고 그 뒤로는 컴파일 타임
상수처럼 취급**되는 값이다. 값이 박히는 순간 드라이버가 셰이더를 다시 최적화한다 —
`if (kOutNhcw)` 같은 분기는 아예 사라진다.

```glsl
// 워크그룹 크기: specialization constant 0/1
layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z = 1) in;
// 출력 레이아웃: specialization constant 2
layout(constant_id = 2) const bool kOutNhcw = false;
```

이 프로젝트는 **워크그룹 크기를 spec constant로 뒀다.** 이유는 단순하다.

> **워크그룹 크기의 최적값은 측정으로만 알 수 있는데, 상수로 박으면 스윕 한 점마다
> 셰이더를 다시 컴파일하고 빌드를 다시 해야 한다.**
> spec constant면 **같은 SPIR-V 하나로 실행 중에 값만 바꿔 가며 20~26조합을 잰다.**

실제로 이 프로젝트의 워크그룹 스윕은 전부 그렇게 했다 —
전처리 [`yvo_preproc_bench --sweep`](../01_src/tools/vulkan/preproc_bench.cpp),
후처리 `yvo_postproc_probe --probe sweep`(26조합), 표시 `yvo_display_probe --probe convert --sweep`(23조합).
`kOutNhcw`(출력 레이아웃), `kUniformScale`(표시 스케일 경로), `kChecksum`(검증 모드),
`kDrawText`/`kTextScale`(오버레이) 도 전부 spec constant다 — **한 벌의 GLSL로 두 가지 셰이더를
만들되 실행 비용은 갈라지지 않게** 하는 수단이다.

---

## 5.5 descriptor set — 셰이더가 볼 버퍼를 미리 묶어 두는 것

셰이더 안의 `binding = 0`, `binding = 1` 은 **"0번 자리에 꽂힌 버퍼"** 라는 뜻이지 특정
버퍼를 가리키지 않는다. 그러면 실제 버퍼는 언제 연결하나?

Vulkan은 이것을 3층으로 나눈다.

```
  VkDescriptorSetLayout   "0번은 storage buffer, 1번도 storage buffer, 2번은 storage image"
        │                  ── 모양(型)만 정의한다. 실제 버퍼는 모른다.
        │                  ── 파이프라인을 만들 때 필요하다.
        ▼
  VkDescriptorPool        디스크립터 셋을 담아 둘 메모리 풀 (미리 크게 잡아 둔다)
        │
        ▼
  VkDescriptorSet         "0번 = 이 Y 버퍼, 1번 = 이 UV 버퍼, 2번 = 이 출력 버퍼"
                          ── 실제 연결. vkUpdateDescriptorSets 로 채운다.
                          ── 커맨드 버퍼에서 vkCmdBindDescriptorSets 로 꽂는다.
```

왜 이렇게 나누나? **미리 묶어 두면 프레임마다 하는 일이 "꽂기" 하나로 줄기 때문**이다.
드라이버 입장에서 "이 셰이더가 어떤 메모리를 건드릴 것인가"를 실행 전에 다 알면
주소 변환·권한 검사를 미리 해 둘 수 있다.

> 이 프로젝트는 그 이점을 끝까지 쓴다. 후처리는 **디스크립터 셋이 슬롯 × 스케일 = 3개/슬롯이고
> 전부 정적**이다 — 초기화 때 한 번만 채우고 프레임 루프에서는 바인딩만 바꾼다
> ([`yolo_postproc.hpp`](../01_src/vulkan/yolo_postproc.hpp) "디스크립터" 절).
> **프레임 루프 안에서 Vulkan 객체를 만들지 않는다**는 것이 이 프로젝트의 규율이다.

---

## 5.6 동기화 — GPU는 알아서 순서를 지키지 않는다

여기가 Vulkan에서 가장 자주 틀리는 곳이다.

### 무엇이 문제인가

커맨드 버퍼에 이렇게 녹음했다고 하자.

```
  vkCmdDispatch(...)   // A: 셰이더가 버퍼 X에 결과를 쓴다
  vkCmdDispatch(...)   // B: 다른 셰이더가 버퍼 X를 읽는다
```

**A가 끝난 뒤에 B가 시작할 것이라고 Vulkan은 보장하지 않는다.** 녹음 순서는 "제출 순서"일
뿐이고, GPU는 성능을 위해 겹쳐 실행할 수 있다. 게다가 A가 쓴 값이 GPU 내부 캐시에만 있고
메모리까지 내려가지 않았을 수도 있다.

이것을 명시하는 수단이 **파이프라인 배리어(pipeline barrier)** 다.

### 배리어가 말하는 두 가지

배리어 하나는 서로 다른 두 가지를 동시에 지정한다. 이 구분을 못 하면 배리어를 영원히 못 쓴다.

| | 이름 | 무엇을 말하는가 | 예 |
|---|---|---|---|
| **언제** | stage mask (`srcStageMask`/`dstStageMask`) | 파이프라인의 **어느 단계**를 기다리고, 어느 단계를 막을 것인가 | `COMPUTE_SHADER`, `HOST`, `ALL_COMMANDS` |
| **무엇을** | access mask (`srcAccessMask`/`dstAccessMask`) | 그 단계의 **어떤 메모리 접근**을 가용(available)·가시(visible)하게 만들 것인가 | `SHADER_STORAGE_WRITE`, `SHADER_STORAGE_READ`, `HOST_READ` |

"A의 컴퓨트 셰이더 쓰기가 끝나고 메모리에 도달한 뒤에야, B의 컴퓨트 셰이더 읽기가 시작되게 하라"는
`src = (COMPUTE_SHADER, SHADER_STORAGE_WRITE)`, `dst = (COMPUTE_SHADER, SHADER_STORAGE_READ)` 이다.

이 프로젝트는 Vulkan 1.3의 **synchronization2**(`VkBufferMemoryBarrier2`,
`vkCmdPipelineBarrier2`)를 쓴다. 구버전보다 마스크를 세밀하게 나눌 수 있고 실수가 줄어든다
([`vulkan_context.hpp`](../01_src/vulkan/vulkan_context.hpp): "Vulkan 1.3 synchronization2 ...
기능을 켠다").

### 배리어가 필요 **없는** 경우도 알아야 한다

배리어를 아무 데나 넣으면 GPU가 매번 멈춰서 느려진다. 이 프로젝트의 후처리가 좋은 예다.

`yolo_decode.comp` 는 스케일 3개에 대해 **3번 디스패치**하는데, **그 사이에 배리어가 없다.**
근거는 셰이더 주석에 적혀 있다:

> "**3 디스패치 사이에 배리어가 없다** — 상호작용은 `det.count` 의 device-scope atomic 뿐이고
> 각 디스패치는 서로 다른 `dets[]` 칸에만 쓴다(주소가 겹치지 않으므로 경쟁이 없다)."

반대로 표시 경로에서는 **필요하다.** `display_convert` 가 쓴 픽셀 위에 `overlay_boxes` 가
박스를 덧그리므로 **같은 픽셀에 쓰기-후-쓰기(WAW, write-after-write)** 가 생긴다
([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 8:
"두 디스패치 사이 `SHADER_STORAGE_WRITE → SHADER_STORAGE_WRITE` 배리어는 **필요하다**").

### fence와 semaphore — 기다리는 두 가지 방식

배리어는 **GPU 안에서의 순서**를 정한다. 그런데 "CPU가 GPU를 기다리는" 것과
"GPU 작업이 다른 GPU 작업을 기다리는" 것은 또 다른 문제다.

| 수단 | 누가 누구를 기다리나 | 이 프로젝트에서 |
|---|---|---|
| **fence(펜스)** | **CPU**가 GPU를 | `vkWaitForFences` 로 전처리 완료를 기다린 뒤 결과를 NPU에 넘긴다 |
| **semaphore(세마포어)** | **GPU**가 GPU를 (큐 사이 / 스왑체인) | 스왑체인 acquire ↔ 렌더 ↔ present 사슬 |

> `vkQueueWaitIdle` 과 `vkDeviceWaitIdle` 은 "GPU가 하던 일 전부 끝날 때까지 CPU가 선다"는
> 극단적인 수단이다. 편하지만 파이프라인을 통째로 비우므로 **초기화·종료 때만** 쓴다.
> 이 프로젝트의 스왑체인은 재생성조차 `vkDeviceWaitIdle` 없이 한다
> ([`vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp) "재생성: vkDeviceWaitIdle 없이").

### queue family ownership — 남의 메모리를 볼 때

디코더나 NPU가 만든 메모리를 GPU가 읽으려면 배리어 하나가 더 필요하다.
**queue family ownership transfer(큐 패밀리 소유권 전환)** 이라고 하고,
자세한 것은 [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) §6.8에서 다룬다.
여기서는 "외부에서 온 메모리는 배리어로 소유권을 가져왔다가 돌려줘야 한다"만 기억하면 된다.

---

## 5.7 timestamp query — GPU 시간을 재는 기본 수단

"이 셰이더가 몇 ms 걸리나"를 CPU 시계(`std::chrono`)로 재면 **틀린다.**
`vkQueueSubmit` 은 곧바로 돌아오고(비동기), 그 뒤 fence를 기다리면 **제출 오버헤드와 대기
시간까지 다 섞여 들어간다.**

GPU 구간의 **시간**을 재는 수단은 **timestamp query**다. 시간이 **왜** 그만큼 걸리는지(연산인지 메모리 대기인지)는
이 절 끝의 **성능 카운터**로 본다.

```
   vkCreateQueryPool(... VK_QUERY_TYPE_TIMESTAMP ...)       // 초기화 때 한 번

   [커맨드 버퍼 녹음]
     vkCmdResetQueryPool(cb, pool, 2*slot, 2);   // 쓰기 전에 반드시 리셋
     vkCmdWriteTimestamp(cb, ..., pool, 2*slot + 0);   ← GPU가 여기 도달한 시각
     vkCmdDispatch(...);
     vkCmdWriteTimestamp(cb, ..., pool, 2*slot + 1);   ← GPU가 여기 도달한 시각

   [완료 후]
     vkGetQueryPoolResults(...)   → 두 값의 차 = tick 수
     ms = tick × timestampPeriod × 1e-6
```

여기서 `timestampPeriod` 가 핵심이다. GPU가 돌려주는 값은 ns도 µs도 아니고 **"틱(tick)"** 이라는
장치마다 다른 단위다. `VkPhysicalDeviceLimits::timestampPeriod` 가 **"틱 1이 몇 나노초인가"** 를
알려준다(vk.xml 1.4.309: "number of nanoseconds it takes for timestamp query value to increment by 1").
이 보드는 1 ns이지만, **코드에 1을 박지 않고 런타임에 읽는다**
([`sand_letterbox.cpp:586-589`](../01_src/vulkan/sand_letterbox.cpp)).

주의할 것이 두 가지 더 있다.

- **`timestamp_valid_bits`**: 카운터가 64비트를 다 쓰지 않을 수 있다. 상위 비트를 마스크해야 한다.
  이 프로젝트 코드는 그렇게 한다.
- **GPU 단독 측정은 NPU가 노는 동안 재야 한다.** NPU를 동시에 돌리면 메모리 대역폭 경합으로
  전처리가 2.14 → 2.20 ms(+3%, V3D 960 MHz 조건)로 밀린다
  ([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) "주의와 남은 일").

> ⚠ 그리고 이 프로젝트에는 **timestamp 평균값을 믿으면 안 되는 조건**이 하나 있다.
> `--display-pace asap` 에 표시를 켠 상태에서는 V3D가 연속 제출된 커맨드 버퍼를 합치면서
> 한쪽 구간이 0에 가깝게 기록되는 현상이 관측됐다(**원인 미확정**). 그래서 이 프로젝트의
> GPU 시간은 **전부 중앙값(median)** 이다 ([`README.md`](../README.md) §5,
> [`07_results/README.md`](../07_results/README.md#-인용-금지-구간) — 정본).

> ⚠ **GPU ms는 V3D 클럭과 함께 읽어야 한다.** 2026-09-25 부트로더 EEPROM이 펌웨어 기본 `v3d_freq`를
> 960 → 1150 MHz로 올렸다. 그 뒤 전처리(`sand_letterbox`)는 b1 2.11 → 1.81 ms, b8 평균 2.6 → 2.0 ms가 됐다
> ([`bootloader_20260925`](../07_results/pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md)).
> 이 장의 GPU ms는 따로 적지 않으면 **960 MHz 조건**의 값이다.

### 시간 말고 "왜"를 보려면 — 성능 카운터(`VK_KHR_performance_query`)

timestamp는 "얼마나"만 알려 준다. V3DV는 `VK_KHR_performance_query`로 하드웨어 카운터 93개(QPU-active,
TMU-active, core-memory-reads, L2T-stalls-read-data 등)를 내준다. 프로브 도구는 `yvo_perf_counters`
([`perf_counters.cpp`](../01_src/tools/vulkan/perf_counters.cpp))이고, 실측으로 확인한 특성은 셋이다
([`v3d_perf_counters`](../07_results/v3d_perf_counters/summary.md) — 정본).

- **한 쿼리는 그 안의 잡을 모두 합산한다.** dispatch 하나만 보려면 쿼리 하나로 그 dispatch만 감싼다.
- **다른 V3D 사용자의 잡이 섞인다.** 데스크톱 렌더가 끼면 우리 카운터가 멈추거나(절단) 남의 값이 들어온다(유입).
  그래서 표시를 끄고, 중앙값을 쓰고, 결정적 카운터(QPU-active)가 최빈값과 다른 표본은 버린다.
- **클럭을 올려도 메모리 대기는 줄지 않는다.** 1150 MHz(클럭 1.198배)에서 4 MiB 합 셰이더는 1.10~1.13배만 빨라졌다.
  연산 사이클은 클럭을 따라 줄고, 메모리 읽기 대기(ns)는 그대로여서 구간의 약 41%가 대기였다(**추정**, 카운터 의미 미검증).
  같은 회차에서 한 번의 시간이 +2.6~6.3% **계단**으로 바뀌는 현상도 보였고 원인은 모른다
  ([`runs/bootloader_20260925`](../07_results/v3d_perf_counters/runs/bootloader_20260925/summary.md)).

---

## 5.8 GLSL → SPIR-V 툴체인과 검증 레이어

### 셰이더는 두 번 번역된다

```
   sand_letterbox.comp        사람이 쓰는 GLSL 소스
        │
        │  glslc  (shaderc)         ← 빌드 시점
        ▼
   sand_letterbox.spv         SPIR-V — GPU용 중간 표현(bytecode)
        │
        │  V3DV 드라이버 (Mesa)      ← 실행 시점(vkCreateComputePipeline)
        ▼
   V3D 기계어
```

**SPIR-V**(스퍼-브이)는 Vulkan이 받아들이는 유일한 셰이더 형식이다. GLSL 텍스트를 런타임에
컴파일하던 OpenGL과 달리, Vulkan은 **미리 컴파일된 바이너리**를 요구한다. 덕분에
문법 오류가 빌드 시점에 잡히고, 실행 시 드라이버가 할 일이 줄어든다.

### 검증 레이어가 잡아 주는 것

**검증 레이어(validation layer)** 는 Vulkan 호출을 가로채서 스펙 위반을 검사하는 계층이다.
기본으로는 꺼져 있고(느려진다), 개발 중에만 켠다.

```bash
VK_LOADER_LAYERS_ENABLE='*validation*' ./yvo_pipeline ...
```

레이어가 위반을 찾으면 **VUID**(Valid Usage ID)라는 고유 번호와 함께 메시지를 낸다.
예: `VUID-VkSwapchainCreateInfoKHR-presentMode-02839`.

VUID가 왜 중요한가? **그 번호로 스펙 원문을 정확히 찾을 수 있기 때문이다.**
추측으로 고치지 말고 원문을 읽고 고치는 것이 이 프로젝트의 규율이다.

> 실제로 이것이 통했다. 표시 경로 개발에서 **검증 레이어가 실제 버그 3개를 잡았다** —
> FIFO `minImageCount=3` 의 VUID 위반, blit 폴백의 WRITE_AFTER_READ, 세마포어 소비가
> 끝나기 전 `vkReleaseSwapchainImagesEXT` 호출
> ([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) 결론 6).
> 특히 첫 번째는 스펙 원문("minImageCount는 `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` 가
> 돌려준 minImageCount 이상이어야 한다", Vulkan 1.4.309)을 읽은 결과
> **"기본값을 안전한 4로 두고, 3은 명시적 opt-in 플래그로 남긴다"** 는 설계 결정으로 이어졌다
> ([`vk_swapchain.hpp`](../01_src/display/vk_swapchain.hpp) `lean_images`).

이 프로젝트의 하드 게이트는 **"검증 레이어 errors 0"** 이다. 경고 1건(장치 스캔 관련)은
알려진 무해 항목이다 ([`README.md`](../README.md) §5).

---

## 5.9 이 프로젝트에서는 — V3D의 한계가 설계를 어떻게 바꿨나

여기서부터가 이 챕터의 본론이다. **"Vulkan 일반론"과 "이 보드에서 실제로 되는 것"은 다르다.**

### V3D(VideoCore VII)의 실제 한계

| 항목 | 값 | 이것이 뭘 막는가 |
|---|---|---|
| `maxComputeWorkGroupInvocations` | **256** | 워크그룹 하나에 invocation 256개까지. `local_size_x*y*z ≤ 256` |
| `maxComputeSharedMemorySize` | 32 KB | shared memory 한도 (`efficient_nms.comp` 만 약 24.9 KB 쓴다. 후보 상한 1024개가 이 한도에서 나왔다) |
| subgroup 크기 | **16** | 분기가 갈라지는 단위 |
| `maxPerStageDescriptorStorageBuffers` | **8** | **한 셰이더가 SSBO를 8개까지만 본다** ← 아래 참조 |
| **`descriptorIndexing`** | **false** | **버퍼를 배열로 두고 인덱스로 고를 수 없다** ← 아래 참조 |
| `shaderStorageImageWriteWithoutFormat` | **false** | storage image에 쓰려면 GLSL에 포맷 한정자 필수 (→ [07_video_display.md](07_video_display.md)). 어기면 에러 없이 틀린다 — FFmpeg DPX compute 디코더가 R 채널만 기록했다 ([`vulkan_compute_codecs`](../07_results/vulkan_compute_codecs/summary.md)) |
| **`shaderInt64`** | **false** | 셰이더에서 64비트 정수를 못 쓴다. 쓰면 셰이더 컴파일러가 abort한다(FFv1 `u2u64`, 같은 정본). 자체 셰이더는 int32로만 짠다 |
| Vulkan Video (`VK_KHR_video_*`) | **확장 0개** | GPU로 영상 디코드를 못 한다. HEVC는 별도 HW 디코더(V4L2)만 쓴다 (같은 정본, [07장](07_video_display.md) §7.2) |
| `R8G8B8A8_UNORM` `linearTilingFeatures` | `0xc980` — **STORAGE_IMAGE 없음** (optimal 은 있음) | 컴퓨트가 LINEAR 이미지에 쓸 수 없다 → HVS 박스 평면(AB24 LINEAR)은 storage buffer 에 그린다 (`overlay_boxes_buf.spv`, [`overlay_plane.hpp`](../01_src/display/overlay_plane.hpp), [07장](07_video_display.md) §7.12) |
| 32비트 정수 곱셈 | **없음** (umul24 에뮬레이션) | 셰이더 안의 곱셈이 비싸다 → 주소 계산을 덧셈 누적으로 |
| 메모리 타입 | `DEVICE_LOCAL｜HOST_VISIBLE｜HOST_COHERENT` **하나뿐** | → [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) 의 주제 |

> 이 표는 관측값이다. 코드가 값에 의존하면 **하드코딩하지 말고 런타임에 질의**하는 것이
> 이 프로젝트의 규율이다(`timestampPeriod`, 헤드룸 질의가 그 예다).

### ★ 6텐서를 한 셰이더에 못 몰아서 3 디스패치로 나눈 것

NPU는 프레임마다 텐서 **6개**를 내놓는다(스케일 3개 × {클래스, 박스}).
자연스러운 설계는 "셰이더 하나가 6개를 다 읽고 한 번에 처리"다.

그런데 못 한다. 이유가 두 개 겹친다.

1. **`maxPerStageDescriptorStorageBuffers = 8`.** 입력 6 + 출력 1 = 7이니 겨우 들어가긴 한다.
2. **`descriptorIndexing = false`.** 이게 결정타다. 이 기능이 있으면 GLSL에서
   `buffer Tensors { ... } tensors[6];` 처럼 **버퍼 배열을 만들고 실행 중에 인덱스로 고를 수** 있다.
   없으면 `binding` 번호가 **컴파일 타임 상수**여야 한다.

결과적으로, 6개를 한 셰이더에 넣으면 스케일마다 분기를 만들 수 없으니
**스케일별 코드를 3벌 펼쳐 써야 한다.** 셰이더 코드가 3배가 되고, 그것은
"로컬 배열+루프를 못 푸는" V3D 컴파일러에게 최악이다.

그래서 채택한 것이 이것이다:

```
   ┌─ 파이프라인은 하나 ─────────────────────────────────────────┐
   │  yolo_decode.comp  (SSBO 3개: cls, box, det)               │
   └────────────────────────────────────────────────────────────┘
        │ push constant만 바꿔서 3번 디스패치
        ├─ dispatch #1: scale_id=0 (P3, 80×80, stride 8)   cls₀ box₀ det
        ├─ dispatch #2: scale_id=1 (P4, 40×40, stride 16)  cls₁ box₁ det
        └─ dispatch #3: scale_id=2 (P5, 20×20, stride 32)  cls₂ box₂ det
                                       배리어 없음 (서로 다른 dets[] 칸에만 쓴다)
```

근거는 [`yolo_decode.comp`](../02_shaders/yolo_decode.comp) 머리 주석에 그대로 적혀 있다:

> "한 디스패치가 보는 SSBO 는 `{cls, box, det}` **3개뿐**이다
> (`maxPerStageDescriptorStorageBuffers = 8`, `descriptorIndexing = false` → 6개를 한 셰이더에
> 몰면 바인딩을 동적으로 못 골라 스케일별 코드를 3벌 펼쳐야 한다. V3D 컴파일러에 최악이므로 기각했다)."

**하드웨어 한계 두 개가 파이프라인 구조를 바꾼 사례**다.

### 셰이더 5개가 하는 일 (한 줄씩)

| 셰이더 | 입력 → 출력 | 하는 일 | 실측 중앙값 |
|---|---|---|---:|
| [`sand_letterbox.comp`](../02_shaders/sand_letterbox.comp) | 디코더 SAND128 NV12 → NHCW 640×640×3 | de-SAND + 축소 + YUV→RGB + letterbox 패딩 | **2.088 ms** |
| [`yolo_decode.comp`](../02_shaders/yolo_decode.comp) | 원시 6텐서 → 후보 박스 ~16개 | dequantize + DFL softmax + threshold + 좌표 역변환 | **0.626 ms** (표의 값은 960 MHz. 1150 MHz 단계 분해(argmax 지름길 이후, 2026-10-11): 클래스 스캔 약 0.34 ms 고정, emit은 통과 클래스가 1개면 클래스 텐서를 다시 읽지 않아 운용 임계값 0.2에서 0.04~0.05 ms(혼잡 장면에서도. 임계값을 0.02·0.003으로 낮추면 폴백이 늘어 0.14·1.09 ms), DFL은 후보 수에 비례, 좌표 역변환은 0.01 ms 미만 — [`postproc_stage_cost`](../07_results/postproc_stage_cost/summary.md), 회차 [`emit_shortcut`](../07_results/postproc_stage_cost/runs/emit_shortcut/summary.md)) |
| [`display_convert.comp`](../02_shaders/display_convert.comp) | 디코더 SAND128 NV12 → 스왑체인 RGBA | de-SAND + YUV→RGB (1:1) | **4.153 ms** |
| [`overlay_boxes.comp`](../02_shaders/overlay_boxes.comp) | 검출 목록 → 스왑체인 RGBA (변형 `overlay_boxes_buf`: → AB24 dma-buf 버퍼) | 박스 테두리 + 라벨 글자 + HUD 그리기 | **0.128 ms** (image 경로) |
| [`efficient_nms.comp`](../02_shaders/efficient_nms.comp) | 후보 박스 → 최종 검출 | bitonic 정렬 + 클래스별 greedy 억제 | **0.049~0.050 ms** (파이프라인 중앙값, 1150 MHz ¹) |

합 **6.995 ms**(`efficient_nms.comp` 추가 전 4개), 프레임 주기 28.6 ms의 24.5%다
([`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.1, `disp_asap`).
표의 ms 값과 아래 워크그룹 스윕은 모두 **V3D 960 MHz** 조건이다. 1150 MHz(부트로더 2026-09-25)에서 전처리는 표시 off 기준
b1 1.81 ms, b4·b8 약 2.0 ms다([`bootloader_20260925`](../07_results/pipeline_ffmpeg_vulkan_hailo/runs/bootloader_20260925/summary.md)). 나머지 셰이더는 1150 MHz에서 다시 재지 않았다.

¹ 정식 A/B [`gpu_nms_e2e`](../07_results/gpu_nms_e2e/summary.md)(2026-10-08, 표시 off, b2s4·b8s16, 폴백 0). 단독 프로브 0.054 ms, 후보 1024 최악 3.05 ms. CPU 억제는 약 2 µs/frame이라 CPU 절감은 보이지 않고 회귀도 없다. 혼잡 장면(후보 평균 약 180개)에서는 0.178~0.186 ms이고, 후보가 1024개를 넘는 프레임은 CPU로 넘긴다([`nms_crowd_stress`](../07_results/nms_crowd_stress/summary.md)).
색 변환식과 letterbox 규칙의 상세는 [07_video_display.md](07_video_display.md),
텐서 해석은 [03_yolo_tensor.md](03_yolo_tensor.md) 에 있다.

### ★ 워크그룹 크기는 측정해서 고른다 — 그리고 규율이 셰이더마다 다르다

"워크그룹은 크게 잡는 게 좋다"는 흔한 조언이다. 이 프로젝트에서 **그 조언이 반은 맞고 반은 틀렸다.**

**전처리** `sand_letterbox` 는 조언대로였다
([`sand_letterbox.hpp`](../01_src/vulkan/sand_letterbox.hpp), 1080p→640 스윕):

```
    64 invocation  →  3.65 ms
   128 invocation  →  2.45 ms
   256 invocation  →  2.13 ms   ← 32×8 채택. 클수록 좋다.
```

**표시 변환** `display_convert` 도 "클수록 좋다"는 맞았지만, **같은 invocation 수 안에서
타일 모양이 1.83배를 갈랐다** ([`display_convert.hpp`](../01_src/display/display_convert.hpp), 23조합):

```
   16×16 (256 inv,  64×16 px 타일)  →  7.74 ms
    3×85 (255 inv,  12×85 px 타일)  →  4.23 ms   ← 채택. 세로로 긴 타일이 결정적으로 빠르다.
```

**⚠ 이유는 미확정이다.** 읽기만 보면 SAND 열이 가로 128 B이므로 좁은 타일이 캐시 라인을
낭비해서 **불리해야 한다.** 그런데도 빠르다. 담당자의 해석은 "병목이 읽기가 아니라 타일 배치(UIF)된
RGBA 이미지 **쓰기** 쪽이고, 세로로 긴 영역이 이미지 타일에 더 잘 맞는다"인데,
헤더 주석이 명시적으로 **"이 해석은 미확정 가설이고, 수치만이 근거다"** 라고 적어 두었다.

**후처리** `yolo_decode` 는 아예 조언이 뒤집혔다
([`yolo_postproc.hpp`](../01_src/vulkan/yolo_postproc.hpp), 26조합):

```
   10×6  =  60 invocation  →  0.566 ms   ← 채택
   16×16 = 256 invocation  →  0.733 ms
   64×4  = 256 invocation  →  0.987 ms
   80×3  = 240 invocation  →  1.269 ms
```

**60 invocation이 256보다 1.3배 빠르다.** 게다가 32·64·128(subgroup 16의 배수)이
20·40·60·120보다 느리다.

**⚠ 이것도 미확정 가설이다.** 헤더 주석의 표현을 그대로 옮기면:

> "총 작업량이 작아(유효 invocation 2100개) 큰 워크그룹으로는 워크그룹 수가 8개까지 떨어져
> 코어에 덜 퍼지는 것으로 **추정**한다(10×6 은 P3 에서 28 워크그룹이다). 미확정 가설이다."
> "③ 32·64·128 invocation(워프 배수)이 20·40·60·120 보다 **느리다**. 설명하지 못했다."

**교훈은 "최적값이 몇인가"가 아니다.** 셰이더마다, 해상도마다 다르고 이유도 모른다는 것이
교훈이다. 그래서 이 프로젝트는 **값을 spec constant로 두고 조건이 바뀌면 다시 잰다.**

### ★ 메모리를 다시 읽는 대신 레지스터에 메모해 두기 — `yolo_decode` ⑤의 argmax 지름길

후처리 pass 1(`yolo_decode.comp`)은 스레드 하나가 **가로로 이웃한 앵커 4개**를 맡아 다섯 단계를 돈다.
앵커는 출력 격자의 한 칸이다(P3 80×80 + P4 40×40 + P5 20×20 = 8400개, [03장](03_yolo_tensor.md)).

```
① 클래스 80개를 훑어 앵커별 최고 점수를 구한다      ← 클래스 텐서 80워드를 메모리에서 읽는다
② 4개 모두 임계값 미만이면 끝 (프레임당 2100그룹 중 기본 영상은 십여 개, 혼잡 장면은 약 110개만 남는다)
③ DFL 디코드 (박스 4변)   ④ 원본 좌표로 되돌리기
⑤ emit — 통과한 (앵커, 클래스)를 dets[]에 쓴다     ← 예전에는 여기서 80워드를 "다시" 읽었다
```

예전 ①은 **최고 점수만** 레지스터에 남겼다. 그래서 ⑤에서 "어느 클래스가 통과했나"를 알려면 같은 80워드를
메모리에서 다시 읽어야 했다(재스캔). 측정해 보니 **⑤ 비용의 94.4~97.8%가 이 재스캔**이었고 atomic 몫은 0이었으며,
통과한 앵커는 **100.00%가 클래스 정확히 1개**였다([`emit_split`](../07_results/postproc_stage_cost/runs/emit_split/summary.md)).
즉 ①이 이미 본 그 한 클래스를 찾으려고 창고에 다시 다녀오고 있었다.

#### 레지스터와 메모리 — V3D 7.1에서

| | 레지스터 | 메모리 |
|---|---|---|
| 위치 | QPU 안. 이 보드는 QPU **12개**(슬라이스 3 × 4) | SoC 밖 DRAM. CPU·NPU와 공유 |
| 개수·크기 | **rf0~rf63**. 하나 = 16레인 × 32비트. 스레드당 **64개**(1·2스레드) / **32개**(4스레드) | GB 단위 |
| 접근 | 명령의 피연산자로 바로 | TMU에 주소를 보내고(`tmua`) 결과를 받는다(`ldtmu`). L2를 거치는 **왕복 대기** |
| 비고 | V3D 7.1에는 4.x의 누산기 r0~r5가 **없다**. rf0은 유니폼 읽기의 암묵 목적지라 할당기가 아껴 둔다 | 레지스터가 모자라면 컴파일러가 값을 메모리로 내린다(spill) |

근거는 Mesa 26.2.2 소스(`src/broadcom/compiler/vir_register_allocate.c`의 `PHYS_COUNT 64`·스레드별 분할 주석,
`src/broadcom/common/v3d_device_info.c`의 `has_accumulators = ver < 71`)와 이 보드의 V3D ident 값이다
([`rescan_causes/env_bench.txt`](../07_results/postproc_stage_cost/runs/rescan_causes/env_bench.txt)).
컴파일러는 4스레드부터 시도하고, 레지스터 32개에 안 들어가면 2스레드로 내린다. `yolo_decode`는 동시에 살아 있는
값이 최대 55개(`V3D_DEBUG=shaderdb`의 `max-temps`, 원시 덤프는 git 밖)라 **2스레드**다.

**왜 80워드를 그냥 레지스터에 다 두지 않았나** — 둘 자리가 없다. 80워드는 스레드당 64개를 넘고,
다른 값들이 이미 최대 55개를 차지한다. 그래서 전부가 아니라 **요약 두 개**만 들고 나온다.

#### 방법 — ①의 같은 루프에서 요약 두 개를 함께 만든다

```glsl
uvec4 pmax4 = uvec4(0u);   // 앵커별 (최고 q << 7) | 그 클래스      — 레지스터 4개
uint  ge_cnt = 0u;         // 바이트 a = 앵커 a에서 q >= t_sw 인 클래스 수 — 레지스터 1개
for (int c = 0; c < pc.num_classes; ++c) {
    const uint word = cls_w[ci];                       // 앵커 4개의 같은 클래스 점수 (바이트 4개)
    pmax4 = max(pmax4, pack4(word, uint(c)));          // 값과 번호를 묶어 max 한 번으로 argmax까지
    ge_cnt += ((((word | 0x80808080u) - t_sw4) | word) & 0x80808080u) >> 7;   // 4개 동시 비교(SWAR)
    ci += Wp;
}
```

([`yolo_decode.comp`](../02_shaders/yolo_decode.comp) ①, `pack4()`)

| 변수 | 비트 배치 | 쓰는 곳 |
|---|---|---|
| `pmax4[a]` | 비트 7~14 = 점수 `q`(8비트), 비트 0~6 = 클래스 `c`(7비트). 점수가 큰 쪽이 이기고 번호가 따라온다 | ② 그룹 탈락(`pmax ≥ q_thr_low << 7`), ⑤ 지름길의 클래스·점수 |
| `ge_cnt` | 32비트 하나를 바이트 4칸으로 나눠 앵커 4개의 개수를 센다(최대 80이라 넘치지 않는다) | ⑤에서 지름길(1) / 재스캔(2 이상) 선택 |

⑤는 이렇게 바뀌었다 — **통과 개수가 1이면 메모를 쓰고, 아니면 예전 코드 그대로 다시 읽는다.**

```glsl
if (((ge_cnt >> (8u * uint(a))) & 0xffu) == 1u) {      // 지름길: 클래스 로드 0번
    const uint c1 = pmax4[a] & 0x7fu;
    const float score = (float(pmax4[a] >> 7) - pc.cls_zp) * pc.cls_scale;
    if (score >= pc.thr) emit(..., score, c1, key_a | c1);
    continue;
}
// 그 밖: 80워드 재스캔 (폴백, 변경 없음)
```

#### 왜 결과가 바이트 단위로 같은가

- 개수는 느슨한 정수 기준 `q ≥ q_thr_low`(`q_thr_low = ⌊thr / scale⌋`)로 센다. 확정 판정 `(q − zp) × scale ≥ thr`을
  통과하는 클래스는 **반드시** 이 기준도 통과한다(필요조건).
- 그러니 개수가 1이면 확정 통과 클래스는 많아야 1개이고, 그 클래스는 다른 모든 클래스보다 점수가 높다 —
  **곧 argmax(`pmax4 & 0x7f`)다.** 동점인 두 클래스는 둘 다 기준을 넘으므로 지름길에 오지 못한다.
- 지름길에서도 확정 판정을 한 번 더 하므로, 기준만 넘고 확정 판정을 못 넘는 앵커는 아무것도 쓰지 않는다.

**틀려도 안전한 쪽으로 틀리게 만든다.** SWAR 비교에는 `t_sw = min(q_thr_low, 128)`을 쓴다(빌림이 옆 바이트로
번지지 않게 하는 조건). 이 클램프가 틀리면 개수를 **더 많이** 세어 폴백으로 보낼 뿐, 지름길을 잘못 열지 않는다.
⚠ 흔한 6연산 "hasless" 꼴 `((w − t) & ~w & 0x80…)`로 "간단히" 바꾸면 안 된다 — 빌림 때문에 개수를 **덜** 세어
2클래스 앵커에 지름길을 열 수 있다(셰이더 주석).

#### 실제 데이터로 한 번 ([`trace_group.txt`](../07_results/postproc_stage_cost/runs/rescan_causes/evidence/trace_group.txt))

기본 영상 frame 0, P3, 행 47, 앵커 w=36~39. 임계값 0.2(`q_thr_low` 57):

```
c=2 (car)  word 0xd00e3204  → 앵커별 q = 4, 50, 14, 208
pmax4  = [0x0202, 0x1902, 0x0702, 0x6802]     0x6802 >> 7 = 208, & 0x7f = 2 (car)
ge_cnt = 0x01000000                           앵커 3만 1개 → 지름길, 점수 0.7217 emit
```

같은 앵커를 임계값 0.02(`q_thr_low` 5)로 보면 truck(c=7, q=8, 점수 0.028)도 통과해 `ge_cnt = 0x02010100`이 되고,
앵커 3만 **폴백**으로 가서 car와 truck을 둘 다 쓴다. SWAR 한 줄(truck 워드 `0x08000200`, 기준 5)은 이렇게 돈다.

```
w | 0x80808080  = 0x88808280   바이트마다 최상위 비트를 세워 빌림이 옆 칸으로 못 가게
  - 0x05050505  = 0x837B7D7B   q >= 5 인 칸만 최상위 비트가 1로 남는다
  | w           = 0x8B7B7F7B   q >= 128 인 칸은 원래 최상위 비트로 통과
  & 0x80808080  = 0x80000000
  >> 7          = 0x01000000   → 앵커 3의 칸에 +1
```

#### 결과

| 항목 | 값 | 정본 |
|---|---|---|
| pass 1, 임계값 0.2 | 기본 **−14.0%**, mot20_02 **−32.7%**, mot20_05 **−36.3%**(범위 비중첩) | [`emit_shortcut`](../07_results/postproc_stage_cost/runs/emit_shortcut/summary.md) |
| ⑤ emit | 0.126 → 0.040 · 0.310 → 0.051 · 0.367 → 0.052 ms | 같은 곳 |
| 대가 | ①의 고정비 **+0.039 ms**(후보 수와 무관, 반복당 QPU 명령 35 → 45) | 같은 곳 |
| 레지스터 | `spills:fills` 0:0 유지, 스레드 2 → 2 | 같은 곳 |
| 종단간(mot20_05 900프레임) | V3D CSD **−10.3~−11.4%**(약 0.29 ms/f), GPU 후처리 −32~−35%, **FPS·지연 불변**, 검출 **바이트 동일** | [`argmax_shortcut_mot20_05`](../07_results/gpu_nms_e2e/runs/argmax_shortcut_mot20_05/summary.md) |
| 폴백 빈도 | 임계값 0.2에서 **0건**(⑤ 진입 3871개 전부 지름길). 0.05·0.02에서 0~16.4%, 대부분 비슷한 클래스(car↔truck, handbag↔backpack. 다만 mot20_05 0.02는 42%이고 겹친 물체가 30%) | [`rescan_causes`](../07_results/postproc_stage_cost/runs/rescan_causes/summary.md) |
| local size | 재스윕 뒤에도 10×6 유지 | [`local_sweep_argmax`](../07_results/gpu_postprocess/runs/local_sweep_argmax/summary.md) |

⚠ 임계값 0.003에서는 `q_thr_low = 0`이라 모든 앵커가 폴백으로 가서 **+3.1% 회귀**한다(운용점이 아니라 고치지 않았다).
⚠ 이득은 V3D 예산의 여유이지 **FPS가 아니다**(처리량은 NPU·PCIe가 정한다).

#### 다른 곳에도 쓰는 원리

1. **다시 읽는 이유가 "그게 무엇이었나"라면, 처음 읽을 때 함께 들고 나온다.** 값과 번호를 한 단어로 묶으면
   (`(q << 7) | c`) 비교 한 번으로 둘 다 얻는다.
2. **전부를 레지스터에 둘 수 없으면 요약만 둔다.** 요약으로 결정하지 못하는 드문 경우에만 원래 경로(폴백)로 간다 —
   정확성은 폴백이 지킨다.
3. **지름길 조건은 틀려도 폴백 쪽으로 틀리게** 짠다. 필요조건으로 세고, 클램프는 과대 추정 방향으로.
4. **순효과로 판정한다.** ①이 +0.039 ms 무거워진 것만 보면 기각할 변경이다. 같은 바이너리 안의 기준선
   (`YVO_DECODE_LEGACY_EMIT`)과 교차 측정해 순이득을 확인했다.
5. CPU 코드에서도 같다 — 같은 데이터를 두 번 훑지 않고 한 번에 여러 결과를 모으기(loop fusion), 누적값을 포인터가
   아니라 지역 변수에 두기. 메모리 왕복을 줄이는 것이지 계산을 줄이는 것이 아니다.

### ★ 오버레이를 그래픽스 파이프라인으로 만들지 않은 이유

"박스를 그린다"는 말을 들으면 당연히 그래픽스 파이프라인(삼각형 래스터화)을 떠올린다.
이 프로젝트는 **안 썼다.** 대신 컴퓨트 셰이더가 픽셀 주소를 계산해서 직접 쓴다 —
일종의 **컴퓨트 래스터라이저**다.

이유는 V3D가 **타일 기반 렌더러(tile-based renderer)** 이기 때문이다.
타일 기반 GPU는 화면을 작은 타일로 쪼개 타일 하나씩 온칩 메모리에서 그린 뒤 메모리에 쓴다.
모바일·임베디드 GPU에서 대역폭을 아끼는 표준 기법이다.

문제는 **이미 컴퓨트가 써 둔 이미지 위에 덧그릴 때** 생긴다. 렌더패스를 `LOAD_OP_LOAD`
(기존 내용을 읽어 들임)로 열면, 타일러가 **프레임버퍼 전체를 읽고 다시 쓴다.**

> "720p RGBA 3.69 MB × 2 ≈ 3.6 ms. 박스 몇 개 그리려고 낼 비용이 아니다.
> 컴퓨트는 scatter write 만 하므로 그 비용이 0 이다."
> — [`overlay_boxes.comp`](../02_shaders/overlay_boxes.comp) 머리 주석

실측으로 컴퓨트 오버레이는 검출 5개에 **0.128~0.196 ms** 다. 렌더패스를 열었다면
그리기를 시작하기도 전에 3.6 ms를 냈을 것이다.

**교훈**: "표준적인 방법"이 이 하드웨어에서 표준적으로 빠른 것은 아니다.

### 빌드 파이프라인 — glslc → spirv-val → C++ 헤더 임베드

셰이더는 빌드 시점에 세 단계를 지난다. [`cmake/YvoShaders.cmake`](../cmake/YvoShaders.cmake) 의
`yvo_add_shaders()` 함수가 그 전부다.

```
   02_shaders/sand_letterbox.comp
        │
        │ ① glslc --target-env=vulkan1.3 -O -Werror -MD -MF <out>.d
        │     · -O      최적화
        │     · -Werror 경고를 에러로 (셰이더 경고 0이 게이트다)
        │     · -MD/-MF #include 의존성 파일 생성 → CMake DEPFILE 로 연결.
        │               공용 헤더를 고치면 그 헤더를 쓰는 셰이더가 다시 빌드된다
        ▼
   build/02_shaders/sand_letterbox.spv
        │
        │ ② spirv-val --target-env vulkan1.3
        │     SPIR-V 자체가 유효한지 검사. 실패하면 빌드가 깨진다
        ▼
        │ ③ cmake/yvo_embed_spirv.cmake
        │     .spv 를 C++ 배열로 바꿔 헤더에 박는다
        ▼
   build/02_shaders/generated/sand_letterbox_spv.h    →  const uint32_t kSandLetterboxSpv[] = {...}
```

③이 왜 필요한가? **실행 파일이 `.spv` 파일을 찾아 헤매지 않게** 하기 위해서다.
셰이더가 바이너리 안에 들어 있으면 실행 위치·설치 경로에 무관하게 항상 같은 셰이더가 돈다.
측정 재현성에 직접 기여한다.

이름 규칙은 `snake_case.comp` → `kCamelCaseSpv` 다 (`sand_letterbox.comp` → `kSandLetterboxSpv`).

**변형(variant)**: 같은 소스를 매크로만 바꿔 한 번 더 컴파일하려면 항목을 `<src>@<out_name>@<DEF1>[,<DEF2>...]` 로 쓴다.
`overlay_boxes.comp@overlay_boxes_buf@YVO_OVERLAY_BUFFER` 가 `-DYVO_OVERLAY_BUFFER` 로 컴파일돼
`overlay_boxes_buf.spv` / `kOverlayBoxesBufSpv` 가 된다(glslc 플래그·spirv-val·DEPFILE 은 그대로 적용).
빈 정의(`X,`)는 configure 에서 막는다 — glslc 는 인자 없는 `-D` 에 에러를 찍고도 **종료 코드 0** 을 돌려주기 때문이다.

---

## 5.10 자가 점검

**Q1. 워크그룹은 크게 잡을수록 빠른가?**

**아니다. 셰이더마다 다르고, 측정해야 안다.**
전처리 `sand_letterbox` 는 64→3.65, 256→2.13 ms로 "클수록 좋다"가 맞았지만
([`sand_letterbox.hpp`](../01_src/vulkan/sand_letterbox.hpp)),
후처리 `yolo_decode` 는 **60 invocation(10×6, 0.566 ms)이 256 invocation(0.733~0.987 ms)보다 빠르다**
([`yolo_postproc.hpp`](../01_src/vulkan/yolo_postproc.hpp) 26조합 스윕).
**이 역전의 이유는 확정되지 않았다** — "작업량이 작아 큰 워크그룹에서는 워크그룹 수가 8개까지
떨어져 코어에 덜 퍼진다"는 것은 담당자의 **미확정 가설**이고 수치만이 근거다.
32·64·128이 20·40·60·120보다 느린 이유는 **설명하지 못했다.**

**Q2. NPU 출력 텐서가 6개인데 왜 셰이더를 3번 디스패치하는가?**

V3D의 `maxPerStageDescriptorStorageBuffers = 8` 과 **`descriptorIndexing = false`** 때문이다.
descriptorIndexing이 없으면 셰이더가 버퍼를 배열로 두고 실행 중에 인덱스로 고를 수 없다
(binding 번호가 컴파일 타임 상수여야 한다). 6개를 한 셰이더에 몰면 스케일별 코드를 3벌 펼쳐야 하고
V3D 컴파일러에 최악이라 기각했다. 그래서 **파이프라인 하나 + SSBO 3개 + push constant만 바꾼
3 디스패치**다. 근거: [`yolo_decode.comp`](../02_shaders/yolo_decode.comp) 머리 주석 "디스패치 규약".

**Q3. 왜 "Vulkan은 메인 스레드에서만" 이라는 규율이 있는가?**

`VkQueue` 가 **외부 동기화 객체**이기 때문이다 — Vulkan이 락을 걸어 주지 않으므로 두 스레드가
같은 큐에 동시에 제출하면 안 된다(vk.xml, Vulkan 1.4.309: `vkQueueSubmit` 의 `VkQueue queue` 가
`externsync="true"`). 이 프로젝트에는 큐가 하나뿐인데 제출하는 곳은 전처리·후처리·표시로 여럿이다.
그래서 NPU 완료 콜백(HailoRT 스레드)은 "끝났다"는 사실만 큐에 넣고, **GPU 제출은 메인 스레드가
`pump()` 에서** 한다. 근거: [`infer_stage.hpp`](../01_src/pipeline/infer_stage.hpp) "스레드 규율".

**Q4. GPU 시간을 `std::chrono` 로 재면 왜 안 되는가? 대신 무엇을 쓰는가?**

`vkQueueSubmit` 은 비동기라서 곧바로 돌아온다. CPU 시계로 재면 제출 오버헤드와 대기 시간이
섞인다. GPU 구간은 **timestamp query**(`vkCmdWriteTimestamp` 두 개를 디스패치 앞뒤에 넣고
차이를 읽음)로만 잴 수 있다. 값의 단위는 장치마다 다른 "틱"이라
`VkPhysicalDeviceLimits::timestampPeriod`(틱 1당 ns)를 곱해야 한다 —
**이 값을 코드에 박지 않고 런타임에 읽는다.**
⚠ 이 프로젝트에서는 `--display-pace asap` + 표시 on 조건의 **평균값을 쓰면 안 된다**
(커맨드 버퍼 병합으로 한쪽 구간이 0에 가깝게 잡힌다, 원인 미확정). **중앙값만 쓴다.**

**Q5. 박스를 그리는 데 왜 그래픽스 파이프라인을 안 쓰고 컴퓨트 셰이더를 썼는가?**

V3D가 타일 기반 렌더러라서, 이미 컴퓨트가 써 둔 이미지 위에 `LOAD_OP_LOAD` 로 렌더패스를 열면
타일러가 **프레임버퍼 전체를 읽고 다시 쓴다**(720p RGBA 3.69 MB × 2 ≈ 3.6 ms).
박스 몇 개 그리자고 낼 비용이 아니다. 컴퓨트는 필요한 픽셀에만 흩뿌려 쓰므로(scatter write)
그 고정비가 0이고, 실측 **0.128 ms** 로 끝난다.
근거: [`overlay_boxes.comp`](../02_shaders/overlay_boxes.comp) 머리 주석 "왜 컴퓨트인가",
[`display_e2e/summary.md`](../07_results/display_e2e/summary.md) §3.1.

**Q6. specialization constant는 push constant와 무엇이 다른가?**

push constant는 **매 디스패치마다** 값을 바꿀 수 있는 작은 인자다(≤128 B).
spec constant는 **파이프라인을 만드는 시점에** 값이 박히고 그 뒤 컴파일 타임 상수가 된다 —
드라이버가 그 값을 알고 셰이더를 다시 최적화하므로 `if (kOutNhcw)` 같은 분기가 **사라진다.**
이 프로젝트가 워크그룹 크기를 spec constant로 둔 이유는, 최적값을 측정으로 찾아야 하는데
상수로 박으면 스윕 한 점마다 재컴파일·재빌드가 필요하기 때문이다.
**같은 SPIR-V 하나로 실행 중에 20~26조합을 잴 수 있다.**

**Q7. `yolo_decode`의 argmax 지름길은 무엇을 줄였는가?**

계산이 아니라 **메모리 왕복**을 줄였다. 예전 ⑤는 통과한 앵커마다 클래스 텐서 80워드를 다시 읽었다.
지금은 ①이 같은 루프에서 `pmax4`(최고 점수 + 클래스)와 `ge_cnt`(느슨한 기준 통과 개수)를 레지스터에 남기고,
개수가 1이면 그 메모로 바로 쓴다. 개수 1이면 그 클래스가 곧 argmax라서 결과가 바이트 단위로 같다.
운용 임계값 0.2에서 pass 1이 14~36% 줄었고 폴백은 0건이었다. 근거: §5.9 "메모리를 다시 읽는 대신 레지스터에 메모해 두기".

---

**다음** → [06_zerocopy_dmabuf.md](06_zerocopy_dmabuf.md) (이 셰이더들이 읽고 쓰는 메모리가
어떻게 CPU·GPU·NPU·디코더 사이를 복사 없이 오가는가 — **이 프로젝트에서 가장 중요한 챕터**)
