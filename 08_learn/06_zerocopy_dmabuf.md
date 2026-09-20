# 06. 메모리와 zero-copy — 복사를 없애는 법

> **선수 지식**: [01_big_picture.md](01_big_picture.md) (zero-copy·병목·슬롯),
> [02_hardware.md](02_hardware.md) (칩 다섯 개),
> [05_vulkan_compute.md](05_vulkan_compute.md) (SSBO·배리어·큐)
> **다 읽으면 읽을 수 있는 것**: [`01_src/vulkan/dmabuf_import.cpp`](../01_src/vulkan/dmabuf_import.cpp),
> [`01_src/vulkan/dmabuf_export.hpp`](../01_src/vulkan/dmabuf_export.hpp),
> [`01_src/common/dma_heap_alloc.hpp`](../01_src/common/dma_heap_alloc.hpp) 의 "3자 일관성 규약",
> [`01_src/vulkan/vk_storage_headroom.hpp`](../01_src/vulkan/vk_storage_headroom.hpp),
> [`07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §4.3

---

## 서론 — 왜 이 챕터가 가장 중요한가

[01_big_picture.md](01_big_picture.md) 에서 이 프로젝트의 성과를 이렇게 적었다:
**FPS는 오르지 않았고, CPU 시간이 3.31 → 2.18 ms/프레임으로 줄었다.**

그 절약의 대부분은 **"복사를 없앤 것"** 과 **"CPU가 GPU 메모리를 읽지 않게 만든 것"** 에서 나왔고,
그 둘의 수단이 이 챕터의 주제다. 더 중요한 것은 **이 프로젝트의 설계 자체가 메모리 제약으로부터
역산됐다**는 점이다 — "셰이더가 NHCW로 직접 출력한다"도, "dma-heap을 도입했지만 기존 버퍼는
하나도 옮기지 않았다"도 전부 **§6.7의 한 가지 사실**에서 나온다.

---

## 6.1 메모리 기초 — 주소가 두 가지라는 것

### 물리 주소와 가상 주소

메모리 칩에는 **물리 주소(physical address)** 가 있다. 0번 바이트, 1번 바이트, … 식이다.

그런데 프로그램이 `int* p` 로 다루는 주소는 물리 주소가 **아니다.**
**가상 주소(virtual address)** 라고 하고, CPU 안의 **MMU**(Memory Management Unit)가
가상 주소를 물리 주소로 번역한다.

```
   프로그램 A                   프로그램 B
   가상주소 0x1000              가상주소 0x1000      ← 같은 숫자인데
        │                            │
        │  MMU가 번역                 │  MMU가 번역
        ▼                            ▼
   물리주소 0x7A000             물리주소 0x2B000      ← 실제로는 다른 곳
```

왜 이렇게 하나? **보호**(남의 메모리를 못 건드린다), **편의**(모든 프로그램이 "내 메모리는
0번지부터"라고 생각할 수 있다), **유연성**(물리적으로 흩어진 메모리를 가상 주소에서는 연속으로
보이게 할 수 있다) 때문이다.

### 페이지와 mmap

번역은 바이트 단위로 하지 않는다. **페이지(page)** 라는 고정 크기 덩어리 단위로 한다.
보통 4096 B인데 **이 보드는 16384 B(16 KB)다**
([`dmaheap_unify/summary.md`](../07_results/dmaheap_unify/summary.md) 환경: `PAGESIZE 16384`).
뒤에 나오는 정렬(alignment) 이야기와 직결되므로 기억해 두자.

`mmap()` 은 **"이 파일(또는 장치)의 내용을 내 가상 주소 공간에 붙여 달라"** 는 시스템 콜이다.
성공하면 포인터를 받고, 그 포인터를 평범한 배열처럼 읽고 쓰면 실제로는 파일/장치가 바뀐다.

```c
void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
//          ↑주소는 커널이 정해라      ↑읽기만  ↑공유   ↑이 fd의 내용을
```

**여러 장치가 같은 물리 메모리를 각자의 가상 주소로 보게 만드는 것이 zero-copy의 본질**이고,
`mmap` 은 그중 "CPU가 보는 창"을 여는 수단이다.

---

## 6.2 DMA — 장치가 CPU를 거치지 않고 메모리를 만지는 것

NPU에 640×640×3 = 1,228,800 바이트를 보낸다고 하자. 방법이 두 가지다.

```
  (a) PIO — CPU 가 나른다
      메모리 ──> CPU 레지스터 ──> 장치     (122만 번 반복. CPU 가 그동안 아무것도 못 한다)

  (b) DMA — 장치가 직접 가져간다
      CPU:  "장치야, 물리주소 0x7A000 부터 1,228,800 바이트 가져가라"   ← 이 한 마디만
      메모리 ══════════> 장치                (DMA 컨트롤러가 직접 옮긴다)
      CPU:  (그동안 다른 일을 한다)
```

**DMA**(Direct Memory Access, 직접 메모리 접근)는 (b)다.
장치가 메모리 버스에 직접 붙어서 CPU를 거치지 않고 읽고 쓴다.

이 프로젝트에서 DMA를 하는 주체는 셋이다.

- **비디오 디코더** — 디코드한 프레임을 메모리에 쓴다
- **V3D GPU** — 셰이더가 읽고 쓰는 것이 전부 DMA다
- **Hailo-8 NPU** — PCIe를 통해 입력을 가져가고 출력을 써 놓는다

세 장치가 전부 DMA를 하니까, **잘만 배치하면 데이터가 메모리에서 한 발짝도 움직이지 않고
장치들 사이를 오갈 수 있다.** 그것이 zero-copy다.

그런데 그 "잘만 배치하면"에 큰 함정이 있다. **캐시**다.

---

## 6.3 캐시와 일관성 — 이 챕터에서 가장 중요한 절

### 캐시가 왜 있는가

CPU는 빠르고 메모리(DRAM)는 느리다. 격차가 백 배쯤 난다. 그래서 CPU 안에 작고 빠른 메모리를
두고 자주 쓰는 데이터의 사본을 넣어 둔다. 이것이 **캐시(cache)** 다.

캐시는 바이트 단위로 동작하지 않고 **캐시 라인(cache line)** 단위로 동작한다(보통 64 B).
바이트 하나를 읽어도 그 주변 64 B가 통째로 캐시에 올라온다.

```
   연속으로 읽으면 (순차 접근)
   ┌────64 B 캐시 라인────┐
   │ ■■■■■■■■■■■■■■■■■■■ │  ← 한 번 가져와서 64개를 다 쓴다. 효율 100%
   └──────────────────────┘

   3바이트마다 하나씩 읽으면 (stride-3 접근)
   ┌────64 B 캐시 라인────┐
   │ ■··■··■··■··■··■··■· │  ← 한 번 가져와서 21개만 쓴다. 나머지는 낭비
   └──────────────────────┘
```

**이 그림이 이 프로젝트 전체를 설명한다.** §6.7에서 다시 나온다.

### 장치가 메모리를 바꾸면 무슨 일이 생기나

캐시는 **CPU 안에** 있다. DMA를 하는 장치는 캐시를 지나가지 않고 메모리를 직접 만진다.
그래서 이런 일이 생긴다.

```
  장치 → CPU 방향                          CPU → 장치 방향
  1. CPU 가 읽는다 → 캐시에 사본이 남는다     1. CPU 가 쓴다 → 캐시에만 있고 메모리엔 아직(write-back)
  2. NPU 가 DMA 로 새 값을 쓴다               2. NPU 가 DMA 로 읽어 간다 → ★ 메모리의 옛날 값
     (메모리는 새 값, 캐시는 옛날 값)         3. 나중에 캐시가 밀려 나간다
  3. CPU 가 다시 읽는다 → ★ 옛날 값             → ★ NPU 가 써 놓은 값을 덮어쓴다
```

이 문제를 **캐시 일관성(cache coherency)** 문제라고 한다.
**증상이 "간헐적으로 틀린 값"이라 디버깅이 거의 불가능하다.**

### 해법 세 가지 — cached / uncached / write-combine

| 매핑 종류 | CPU 읽기 | CPU 쓰기 | 장치와의 일관성 |
|---|---|---|---|
| **cached** (캐시 사용) | 빠름 | 빠름 | **관리가 필요하다** (명시적 flush/invalidate) |
| **uncached** (캐시 끔) | **느림** — 매번 메모리까지 간다 | 느림 | 자동으로 맞는다 |
| **write-combine** | **느림** (uncached와 같음) | 빠름 — 쓰기를 모아서 한 번에 보낸다 | 쓰기는 버퍼를 비워야(flush) 보인다 |

Vulkan은 메모리 타입에 이것을 플래그로 표현한다 —
`HOST_VISIBLE`(CPU가 `vkMapMemory` 로 볼 수 있다), **`HOST_CACHED`**(호스트에서 캐시된다),
`HOST_COHERENT`(flush/invalidate를 부를 필요가 없다), `DEVICE_LOCAL`(장치 접근에 가장 효율적이다).

Vulkan 스펙(1.4.309, `chapters/memory.html`)이 `HOST_CACHED` 를 이렇게 설명한다:

> "VK_MEMORY_PROPERTY_HOST_CACHED_BIT bit specifies that memory allocated with this type is
> cached on the host. **Host memory accesses to uncached memory are slower than to cached memory**,
> however uncached memory is always host coherent."

### ★ 읽기와 쓰기의 비대칭 — 직관에 반하는 사실

uncached나 write-combine 매핑에서 **쓰기와 읽기의 비용이 전혀 다르다.**

- **쓰기는 싸다.** CPU가 쓰기를 버퍼에 던지고 바로 다음 일을 한다(fire-and-forget).
  하드웨어가 알아서 모아서 메모리로 보낸다.
- **읽기는 비싸다.** CPU가 값이 필요하니까 **결과가 올 때까지 멈춰 서서 기다려야 한다.**
  그리고 캐시가 없으니 다음 바이트를 또 기다린다.

이 비대칭이 이 프로젝트의 설계를 지배한다. **"GPU가 준 메모리에 CPU가 쓰는 것은 괜찮고,
CPU가 읽는 것은 매우 비싸다."**

---

## 6.4 dma-buf — 커널이 주는 "버퍼 공유 증서"

이제 문제를 풀 수 있다. 여러 장치가 같은 물리 메모리를 보게 하려면 **누가 소유하는지 관리할
주체**(→ 커널)와 **드라이버·프로세스 사이로 주고받을 수 있는 표현**이 필요하다.
물리 주소를 직접 주고받으면 안 된다(보안 구멍이고, 페이지가 흩어져 있을 수도 있다).
그래서 리눅스는 **파일 디스크립터(fd)** 로 준다. 이것이 **dma-buf** 다.

### 왜 하필 fd인가

fd는 리눅스에서 이미 잘 동작하는 **"자원에 대한 참조"** 표현이다. 그래서 공짜로 따라오는 것이 많다.

| fd의 성질 | dma-buf에 주는 이점 |
|---|---|
| 참조 계수(reference count)가 있다 | `dup()` 하면 참조가 늘고 `close()` 하면 준다. **마지막 참조가 닫힐 때 버퍼가 해제된다** |
| 프로세스 사이로 넘길 수 있다 (`SCM_RIGHTS`) | 프로세스가 달라도 버퍼를 공유할 수 있다 |
| 권한 검사·`mmap()`·`ioctl()` 이 이미 있다 | 남의 메모리를 못 훔쳐보고, CPU가 보고 싶으면 매핑하면 되고, 캐시 동기화(§6.10)를 ioctl로 붙일 수 있다 |

> 이 프로젝트가 fd 참조 계수에 실제로 기대는 곳:
> `vkAllocateMemory` 가 성공하면 **fd 소유권이 Vulkan으로 넘어가므로**, 우리는 `dup()` 한 사본을
> 넘기고 원본을 계속 들고 있는다. 반대로 실패하면 우리가 `close()` 한다
> ([`dmabuf_import.cpp`](../01_src/vulkan/dmabuf_import.cpp)).

### dma-buf fd가 나오는 곳 두 군데

**(1) DRM_PRIME** — 디스플레이/GPU/코덱 드라이버(DRM 서브시스템)가 만든 버퍼를 dma-buf fd로
내보내는 표준 메커니즘이다. 이 프로젝트에서 **HW 디코더가 이 방식으로 프레임을 준다.**
libavcodec의 `AV_PIX_FMT_DRM_PRIME` 픽셀 포맷이 그것이고, 디스크립터 안에 fd와 오프셋이 들어 있다.

**(2) dma-heap** — 애플리케이션이 **직접 dma-buf를 할당**하고 싶을 때 쓰는 장치 파일이다.
`/dev/dma_heap/system` 같은 경로를 열고 `ioctl(DMA_HEAP_IOCTL_ALLOC)` 을 하면 fd가 나온다.
"어느 장치의 버퍼"가 아니라 **범용 공유 버퍼**를 만드는 수단이다.
이 프로젝트에서는 [`dma_heap_alloc.hpp`](../01_src/common/dma_heap_alloc.hpp) 가 감싸고 있다.

> 둘의 결정적 차이: **dma-heap 버퍼는 CPU에서 캐시된 매핑을 받는다.**
> V3D가 준 버퍼는 그렇지 않다. 이 차이가 §6.11에서 다시 나온다.

---

## 6.5 Vulkan external memory — 남의 메모리를 Vulkan으로 끌어오기

Vulkan은 기본적으로 **자기가 할당한 메모리만** 다룬다. 남이 만든 dma-buf를 쓰려면
확장(extension) 두 개가 필요하다.

| 확장 | 하는 일 | 의존 |
|---|---|---|
| `VK_KHR_external_memory_fd` | 메모리를 **fd로** import/export 하는 일반 틀 | `VK_KHR_external_memory` 또는 Vulkan 1.1 |
| `VK_EXT_external_memory_dma_buf` | 그 fd가 **dma-buf** 일 수 있게 한다 (`VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT` 추가) | `VK_KHR_external_memory_fd` |

(확장 이름과 의존 관계는 vk.xml, Vulkan 1.4.309에서 확인했다.)

이 프로젝트는 둘 다 **필수 확장**으로 요구한다
([`vulkan_context.hpp`](../01_src/vulkan/vulkan_context.hpp): "필수 확장: VK_KHR_external_memory_fd,
VK_EXT_external_memory_dma_buf (dma-buf import/export)").

### import 절차 — 그리고 왜 이 순서인가

[`dmabuf_import.cpp`](../01_src/vulkan/dmabuf_import.cpp) 의 실제 순서다.

```
  ① vkGetMemoryFdPropertiesKHR(dev, DMA_BUF, fd, &fdp)
        → fdp.memoryTypeBits :  "이 fd는 어떤 메모리 타입으로 import 할 수 있는가"

  ② VkBufferCreateInfo{ pNext = VkExternalMemoryBufferCreateInfo{ DMA_BUF } }
     vkCreateBuffer  →  vkGetBufferMemoryRequirements2
        → req.memoryTypeBits :  "이 버퍼는 어떤 메모리 타입에 놓일 수 있는가"
        → req.size          :  "이 버퍼가 실제로 요구하는 크기"  ★ §6.12의 함정

  ③ 메모리 타입 = (① ∩ ②) 중에서 DEVICE_LOCAL 을 선호해 고른다

  ④ dfd = dup(fd)
     VkMemoryAllocateInfo{ pNext = VkImportMemoryFdInfoKHR{ DMA_BUF, dfd } }
     vkAllocateMemory  →  성공하면 dfd 의 소유권이 Vulkan 으로 넘어간다
                          실패하면 우리가 close(dfd) 한다

  ⑤ vkBindBufferMemory(dev, buffer, memory, 0)
```

**왜 이 순서여야 하는가.**

**①이 먼저인 이유**: Vulkan은 dma-buf를 아무 메모리 타입에나 붙일 수 없고, 어떤 타입으로
import할 수 있는지는 **그 fd에 달려 있다**(어느 드라이버가 만들었는지, 어떤 물리 메모리인지).
그래서 커널에 먼저 물어봐야 한다. `VkMemoryFdPropertiesKHR` 의 멤버가 정확히
`memoryTypeBits` 하나인 것이 그 뜻이다(vk.xml, 1.4.309).

**②에서 `VkExternalMemoryBufferCreateInfo` 를 붙이는 이유**: 버퍼를 만들 때
**"이 버퍼는 외부 메모리를 받을 것이다"** 라고 **미리** 선언해야 한다 — 드라이버가 그에 맞춰
내부 배치를 바꿀 수 있기 때문이고, 나중에 붙일 수 없다.
멤버는 `sType`/`pNext`/`handleTypes` 셋뿐이다(vk.xml, 1.4.309).

**③에서 교집합을 쓰는 이유**: ①은 "fd 쪽 제약", ②는 "버퍼 쪽 제약"이고 둘 다 만족해야 한다.
교집합이 비면 import가 불가능하고, 이 프로젝트는 그때 예외를 던진다.

**④에서 `dup()` 하는 이유**: 스펙이 이렇게 말하기 때문이다(Vulkan 1.4.309,
`chapters/memory.html`, File Descriptor External Memory):

> "Importing memory from a file descriptor **transfers ownership of the file descriptor
> from the application to the Vulkan implementation.** The application must not perform
> any operations on the file descriptor after a successful import."

즉 **성공하면 그 fd는 더 이상 우리 것이 아니다.** 원본을 계속 쓰고 싶으므로 사본(`dup`)을
넘긴다. `vkFreeMemory` 를 하면 그 사본도 함께 닫힌다.

**`VkMemoryDedicatedAllocateInfo` 는 언제 붙이나**: "이 할당은 오직 이 버퍼(또는 이미지)
하나만을 위한 것"이라고 선언하는 구조체다(멤버: `sType`/`pNext`/`image`/`buffer`, vk.xml 1.4.309).
드라이버가 요구할 수도 있으므로 `VkMemoryDedicatedRequirements` 로 물어보고
`requiresDedicatedAllocation` 이나 `prefersDedicatedAllocation` 이 서면 붙인다.

> ⚠ 전용 할당을 붙이면 **스펙이 `allocationSize == req.size` 를 강제한다**
> (`VUID-VkMemoryDedicatedAllocateInfo-buffer-02965`, validusage.json, Vulkan 1.4.309:
> "If buffer is not VK_NULL_HANDLE ... VkMemoryAllocateInfo::allocationSize must equal the
> VkMemoryRequirements::size of the buffer"). 이 제약이 §6.12의 헤드룸 해법과 충돌한다 —
> 다행히 **V3D는 6/6 요구하지 않았다**(실측).

### export 절차 — Vulkan 메모리를 남에게 주기

반대 방향은 [`dmabuf_export.hpp`](../01_src/vulkan/dmabuf_export.hpp) 다.

```
  ① VkBufferCreateInfo{ pNext = VkExternalMemoryBufferCreateInfo{ DMA_BUF } }
        → import 때와 같다. "이 버퍼는 외부와 공유될 것이다"를 생성 시점에 선언한다

  ② VkMemoryAllocateInfo{ pNext = VkExportMemoryAllocateInfo{ DMA_BUF } }
     vkAllocateMemory
        → "이 할당은 나중에 fd 로 내보낼 수 있어야 한다"

  ③ vkBindBufferMemory(offset = 0)   ★ 오프셋 0 이라 dma-buf 시작 = 버퍼 시작
  ④ vkMapMemory                      (영구 매핑. CPU 디버깅·픽스처용)
  ⑤ vkGetMemoryFdKHR                 → fd 획득. 이 fd 는 우리 것이다(우리가 close 한다)
```

`VkExportMemoryAllocateInfo` 의 멤버도 `sType`/`pNext`/`handleTypes` 셋이다(vk.xml, 1.4.309).
import 쪽의 `VkImportMemoryFdInfoKHR` 는 `handleType`(**단수** — 하나만 고른다)과 `fd` 를 갖는다.
복수/단수 차이가 의미를 그대로 반영한다: export는 "여러 형식으로 나갈 수 있게",
import는 "이 fd는 이 형식이다".

③의 `offset = 0` 이 중요하다. dma-buf에는 "이 버퍼의 어디부터"라는 오프셋 인자를 받는 API가
드물다(HailoRT의 `read_async` 에도 없다). 그래서 **할당 하나 = 버퍼 하나 = dma-buf 하나**로
1:1 대응시킨다.

---

## 6.6 queue family ownership transfer — 왜 배리어가 하나 더 필요한가

[05_vulkan_compute.md](05_vulkan_compute.md) §5.6에서 배리어를 다뤘다. 외부 메모리에는
배리어가 **한 겹 더** 필요하다.

Vulkan은 "이 자원은 지금 어느 큐 패밀리의 소유인가"를 추적한다. 큐 패밀리마다 캐시 구조와
메모리 접근 방식이 다를 수 있어서다. 소유자가 바뀔 때는 **양쪽이 명시적으로 넘겨줘야** 한다.
이것이 **queue family ownership transfer(큐 패밀리 소유권 전환)** 이고
Vulkan 스펙의 동기화 챕터에 별도 절로 있다(`chapters/synchronization.html#synchronization-queue-transfers`).

그런데 디코더나 NPU는 **Vulkan 큐가 아니다.** Vulkan은 그것들을 위해
특수 큐 패밀리 상수 두 개를 둔다(vk.xml, Vulkan 1.4.309).

| 상수 | 값 | 뜻 |
|---|---|---|
| `VK_QUEUE_FAMILY_EXTERNAL` | `~1U` | **같은 API·호환 장치의** 외부 큐 |
| `VK_QUEUE_FAMILY_FOREIGN_EXT` | `~2U` | **완전히 다른 종류의 장치** (디코더, NPU, 카메라 …) — `VK_EXT_queue_family_foreign` 확장 |

`VUID-VkBufferMemoryBarrier2-buffer-09095`(validusage.json, Vulkan 1.4.309)가 이렇게 못 박는다:

> "If buffer was created with a sharing mode of VK_SHARING_MODE_EXCLUSIVE, and srcQueueFamilyIndex
> and dstQueueFamilyIndex are not equal, srcQueueFamilyIndex **must be VK_QUEUE_FAMILY_EXTERNAL,
> VK_QUEUE_FAMILY_FOREIGN_EXT, or a valid queue family**"

### 실제 모양

```
   [커맨드 버퍼 시작]
     acquire 배리어 :  src = VK_QUEUE_FAMILY_FOREIGN_EXT → dst = 우리 컴퓨트 큐 패밀리
                                                            "디코더한테서 받아온다"
     vkCmdDispatch(...)        ← 셰이더가 그 메모리를 읽는다
     release 배리어 :  src = 우리 큐 패밀리 → dst = VK_QUEUE_FAMILY_FOREIGN_EXT
                                                            "디코더한테 돌려준다"
   [커맨드 버퍼 끝]
```

이 프로젝트는 확장이 있으면 `FOREIGN_EXT`, 없으면 `EXTERNAL` 을 쓴다
([`vulkan_context.hpp`](../01_src/vulkan/vulkan_context.hpp): "선택 확장: VK_EXT_queue_family_foreign
(있으면 외부 소유권 전환에 FOREIGN, 없으면 EXTERNAL)"). 실제 코드는
[`sand_letterbox.cpp:364-386`](../01_src/vulkan/sand_letterbox.cpp)(acquire) 와 `:402-444`(release)다.

### ⚠ "배리어를 빼도 결과가 맞더라"는 배리어가 불필요하다는 뜻이 아니다

후처리 프로브에서 배리어를 아예 끄고(`YoloInputSync::None`) 돌려 봤는데
**틀린 결과가 나오지 않았다** ([`gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md) P3 ④).

그런데도 배리어는 유지한다. 그 문서가 이유를 명시한다:

> "**이것은 배리어가 불필요하다는 증거가 아니다** — 큐 패밀리 소유권 전환은 스펙 요구사항이고
> (VUID-VkBufferMemoryBarrier2-buffer-09095/09096), V3D 의 유일한 메모리 타입이 `HOST_COHERENT` 라
> 이 하드웨어에 스테일 경로가 없는 것으로 **추정**할 뿐이다. 배리어는 유지한다."

**"우연히 동작한다"와 "규약을 지킨다"는 다르다.** 드라이버가 바뀌면 조용히 깨진다.

---

## 6.7 이 프로젝트에서는 — 지배하는 사실 하나

### ★ V3D는 메모리 타입이 하나뿐이고, 그것이 `HOST_CACHED` 가 아니다

Vulkan에서 GPU에게 메모리 타입 목록을 물어보면(`vkGetPhysicalDeviceMemoryProperties`)
보통은 여러 개가 나온다 — "장치 전용", "CPU가 보이는 것", "CPU가 캐시하는 것" 등.

**V3D는 하나만 준다.**

```
   memoryType[0] = DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT
                                                 ↑
                                  HOST_CACHED 가 없다 ★
```

(근거: [`pipeline_ffmpeg_vulkan_hailo/REPORT.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) §2.4,
[`dmaheap_unify/summary.md`](../07_results/dmaheap_unify/summary.md) 1단계 "`memoryTypeBits = 0x1` 6/6")

이것은 UMA(통합 메모리) 구조라서 자연스럽다 — CPU와 GPU가 같은 DRAM을 쓰니까 "장치 전용"
메모리가 따로 없다. 그런데 **`HOST_CACHED` 가 없다**는 것이 문제다.
§6.3의 표로 돌아가면 이 메모리는 CPU 쪽에서 **uncached 또는 write-combine** 이고,
§6.3의 읽기·쓰기 비대칭이 그대로 나타난다.

### 실측 — CPU가 V3D 메모리를 읽는 비용

같은 1,228,800 바이트를 읽는 데:

| 접근 패턴 | 시간 | 대역폭 |
|---|---:|---:|
| **순차** memcpy (V3D 버퍼) | 0.826 ms (P2) / 0.85~0.87 ms (preproc_bench) | **1.479** / 약 1.41 GB/s |
| **stride-3** 바이트 읽기 (V3D 버퍼) | 약 14 ms | 약 **0.088 GB/s** |

**같은 메모리, 같은 양인데 16배 차이다.**
근거: [`pipeline_ffmpeg_vulkan_hailo/summary.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/summary.md) "주의와 남은 일".

왜 그런가? §6.3의 캐시 라인 그림이다. 캐시가 있으면 stride-3도 한 번 가져온 라인에서
21번을 쓸 수 있다. **캐시가 없으면 3바이트마다 매번 메모리까지 간다.**

그리고 결정적인 대조군이 있다. **같은 stride-3 연산을 평범한 `malloc` 버퍼에서 하면 약 1.2 ms다**
([REPORT.md §4.3](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md)).
malloc 메모리는 캐시되니까.

> **"GPU가 쓴 버퍼를 CPU로 훑는 모든 작업은 겉보기보다 약 10배 비싸다.
>  반대로 GPU나 DMA가 그 메모리를 소비하면 비용이 없다."**
> — REPORT.md §4.3 결론 5 (일반화, "다음 작업에 중요")

### ★ 그래서 셰이더가 NHCW로 직접 출력하게 됐다

이 사실이 발견된 경위가 이 프로젝트에서 가장 교훈적인 대목이다.

**증상**: 파이프라인의 CPU 비용이 프레임당 **17.6 ms** 였다. 디코드는 0.25 ms, GPU 제출은
0.15 ms인데 나머지가 어디로 갔는지 몰랐다.

**추적**: `perf` 로 프로파일하니 `libhailort` 의 한 주소(`0x1c10cc` 부근)가 **78.6%** 를 먹었다.
`gdb` 로 중단점을 걸어 콜스택을 보니 `hailort::InputTransformContext::transform` — 채널 3, 폭 640의
**NHWC → NHCW 전치(transpose)** 였다. YOLO 모델의 장치 입력 형식은 **NHCW**(행마다 R 640개,
G 640개, B 640개)인데 우리 셰이더가 **NHWC**(RGB RGB RGB …)로 내놓고 있었고, HailoRT가 그 차이를
**CPU에서** 메워 주고 있었던 것이다.

**첫 번째 가설**: "전치 연산이 비싸구나. 없애자."
→ 셰이더가 NHCW로 직접 쓰게 바꿨다. CPU가 **17.6 → 3.5 ms/frame** 이 됐다.

**그런데 가설이 틀렸다.** 검증해 보니 **전치 연산 자체는 약 1.2 ms** 였다(malloc 버퍼 기준).
파이프라인에서는 14 ms였다. **진짜 원인**은 그 전치 루프가 **1바이트씩 stride 3으로 V3D 메모리를
읽고 있었다**는 것이다. 비싼 것은 전치가 아니라 **"CPU가 V3D 메모리를 최악의 패턴으로 읽는 것"** 이었다.

> "즉 NHCW의 이득은 '전치를 없앤 것'보다 **'GPU가 쓴 메모리를 CPU가 읽는 일을 없앤 것'** 이 본질이다."
> — REPORT.md §4.3 근거 4

**교훈 둘**: ① 고쳐서 빨라졌다고 원인을 맞힌 것은 아니다 — **원인을 따로 검증해야 한다.**
② 이 보드에서 CPU를 절약하려면 **"CPU가 GPU 메모리를 읽지 않게"** 설계해야 한다.
이 교훈이 그 뒤 모든 결정의 기준이 됐다 — GPU 후처리를 도입할 때도, dma-heap을 검토할 때도,
표시 경로를 설계할 때도.

---

## 6.8 dma-buf 세 그룹 — 실제로 무엇이 어디로 오가나

이 프로젝트에는 dma-buf가 **세 무리** 있다. 각각 만든 주체와 소비자가 다르다.

```
 ╔═ 그룹 ① 디코더 SAND ═══════════════════════════════════════════════════════╗
 ║  만든 곳 : HW HEVC 디코더 (libavcodec DRM_PRIME)                            ║
 ║  개수    : 프레임당 2개 (Y 평면용, UV 평면용 — 서로 다른 dma-buf 객체)        ║
 ║  크기    : Y 2,097,152 B / UV 1,048,576 B (1920×1088 SAND128 NV12)          ║
 ║  소비자  : ② sand_letterbox.comp  (추론용 축소)                             ║
 ║            ⑥ display_convert.comp (화면 표시용 1:1 변환)  ← ★ 같은 버퍼를 둘이 읽는다 ║
 ║  Vulkan  : import (DmaBufImportCache, fd 번호로 캐시)                        ║
 ╚═════════════════════════════════════════════════════════════════════════════╝
            │  복사 0                            │  복사 0
            ▼                                    ▼
 ╔═ 그룹 ② 전처리 출력 NHCW ══════════════════════════════════════════════════╗
 ║  Vulkan(DmaBufExporter) 이 만든다. 슬롯당 1개, 640×640×3 = 1,228,800 B      ║
 ║  소비자 : Hailo-8 NPU (H2D DMA).  export → fd → HailoRT                    ║
 ╚═════════════════════════════════════════════════════════════════════════════╝
            │  복사 0                    (표시 경로는 위에서 갈라진다 → 07장)
            ▼
 ╔═ 그룹 ③ 원시 출력 텐서 ═════════════════════════════════════════════════════╗
 ║  Vulkan(DmaBufExporter, 16384 B 정렬)이 만든다. 슬롯당 6개(스케일 3 × 2),    ║
 ║  합 1,221,120 B.  우리가 할당했으므로 import 불필요(export만)                ║
 ║  쓰는 쪽 : Hailo-8 NPU (D2H DMA) — NPU 가 결과를 여기에 써 넣는다            ║
 ║  읽는 쪽 : yolo_decode.comp (V3D GPU)                                       ║
 ╚═════════════════════════════════════════════════════════════════════════════╝
            │  GPU 가 압축한 후보 ~16개 (≈1.6 KB)
            ▼
        CPU (cpu_nms_greedy_by_class)  ← ★ CPU 가 읽는 것은 여기뿐이고, 1.6 KB 라 무해하다
```

**소유 방향이 그룹마다 다르다는 것**을 보자.

- 그룹 ①은 **디코더가 만들고 우리가 import 한다.** 수명은 디코더 것이다 —
  `DrmFrame.keepalive` 를 놓는 순간 디코더 풀로 반납된다.
- 그룹 ②③은 **우리(Vulkan)가 만들고 export 해서 NPU에게 준다.**
  소유권이 우리에게 있으니 정렬·크기를 우리가 통제할 수 있다 — 이것이 §6.12·§6.13에서 중요해진다.

**그리고 CPU가 큰 버퍼를 읽는 곳이 하나도 없다.** §6.7의 교훈이 구조에 새겨진 결과다.
CPU가 읽는 것은 GPU가 압축해 준 후보 목록 약 1.6 KB뿐이고,
그 비용은 **1.1~4.0 µs** 다 ([`gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md) 결론 8).

---

## 6.9 zero-copy는 `dmabuf + NHCW` 에서만 성립한다

이 프로젝트의 실행 옵션에 `--mode ptr|dmabuf` 와 `--layout nhwc|nhcw` 가 있다.
네 조합 중 **하나만** zero-copy다. 이유를 조합별로 보자.

| `--mode` | `--layout` | 무슨 일이 일어나나 | CPU ms/frame |
|---|---|---|---:|
| `ptr` | `nhwc` | HailoRT가 V3D 메모리를 CPU로 읽어 전치·복사 | **17.6** |
| `dmabuf` | `nhwc` | **똑같다.** dma-buf로 줘도 HailoRT가 CPU 전치를 하므로 아무것도 절약되지 않는다 | **17.6** |
| `ptr` | `nhcw` | **시작할 때 거부된다** ← 아래 | – |
| **`dmabuf`** | **`nhcw`** | HailoRT가 버퍼를 **그대로 DMA로 가져간다.** CPU가 안 건드린다 | **3.5** |

(근거: [REPORT.md §4.3](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) 근거 2·3)

`dmabuf + nhwc` 가 `ptr + nhwc` 와 똑같다는 것이 중요하다.
**"dma-buf를 쓰면 zero-copy"가 아니다.** 소비자가 그 버퍼를 **CPU로 건드리지 않아야** zero-copy다.

### ★ 왜 `ptr + nhcw` 는 불가능한가

NHCW로 넘기면 HailoRT는 CPU 변환을 건너뛰고 **버퍼를 직접 DMA 매핑**한다.
그런데 V3D가 준 포인터(`vkMapMemory` 결과 = GEM mmap 포인터)는
**Hailo 드라이버가 DMA 매핑하지 못한다**
(`Ioctl HAILO_VDMA_BUFFER_MAP failed due to invalid address`).
그래서 NHCW는 **dma-buf로만** 넘길 수 있고, 프로그램은 `--mode ptr --layout nhcw` 를
**시작 시점에 거부**한다.

> 흥미로운 대조: 실패하는 것은 **V3D 포인터**이지 호스트 포인터 일반이 아니다.
> 평범한 `malloc`(페이지 정렬) 버퍼는 NHCW에서도 DMA 매핑에 **성공**한다
> (REPORT.md §4.3 근거 6). 러너는 포인터 출처를 알 수 없으므로
> `allow_host_ptr_nhcw`(기본 false)를 호출자가 명시적으로 켜야 한다.

---

## 6.10 `DMA_BUF_IOCTL_SYNC` — 캐시 동기화의 계약

dma-buf를 CPU가 `mmap` 해서 읽거나 쓸 때는, 커널에 **"지금부터 CPU가 접근한다 / 끝났다"** 를
알려야 한다. 그 수단이 `ioctl(fd, DMA_BUF_IOCTL_SYNC, ...)` 이고, 플래그가 네 개 있다.

```
   START｜READ   "읽을 거다" → 커널이 캐시를 invalidate  … CPU 읽기 …  END｜READ   "다 읽었다"
   START｜WRITE  "쓸 거다"                               … CPU 쓰기 …  END｜WRITE  "다 썼다"
                                                                      → 커널이 캐시를 clean(flush)
```

### ★ 배리어와 sync ioctl은 대체재가 아니다

이 구분이 이 챕터에서 가장 헷갈리는 지점이다.
[`dma_heap_alloc.hpp`](../01_src/common/dma_heap_alloc.hpp) 가 한 문장으로 정리한다:

> "`DMA_BUF_IOCTL_SYNC` 는 **CPU 캐시만** 다루고 순서를 만들지 않는다.
>  Vulkan 배리어는 **V3D 캐시만** 다루고 CPU 캐시를 건드리지 않는다. 둘은 대체재가 아니다."

```
              순서를 만든다   CPU 캐시   V3D 캐시
   Vulkan 배리어      ✔          ✘         ✔
   DMA_BUF SYNC       ✘          ✔         ✘
```

그래서 같은 헤더에 **3자 일관성 규약 R1~R6** 이 글로 적혀 있다:

| 규칙 | 내용 |
|---|---|
| R1 | 장치↔장치(Hailo DMA 쓰기 → V3D 읽기): **sync ioctl 없음.** Vulkan 배리어만 |
| R2 | CPU 읽기: 완료 신호 **뒤에** START｜READ → 읽기 → END｜READ |
| R3 | CPU 쓰기: START｜WRITE … END｜WRITE. **초기화 `memset` 도 예외가 아니다** |
| R4 | 읽고 쓰면 READ｜WRITE. 방향 비트를 아끼지 않는다 |
| R5 | strict 모드에서 ioctl 실패는 예외다 |
| R6 | 비용: 1.2 MB당 START+END 합 약 0.057 ms. 고정비가 크므로 작은 버퍼에는 손해다 |

R3이 왜 예외가 아닌가? 실제로 이 프로젝트에 그 결함이 있었다 — `HeapOutPool` 의
sync 없는 `memset` 이다. 더러운 캐시 라인이 나중에 밀려 나가면서 **장치가 써 놓은 데이터를
덮어쓴다.** 비결정적이라 디버깅이 거의 불가능하다
([`dmaheap_unify/summary.md`](../07_results/dmaheap_unify/summary.md) 결론 2).

### ★ sync가 정말 필요한지 실증에 실패했다 — 그래도 유지한다

이것이 이 프로젝트에서 가장 성숙한 판단 중 하나다.

[`dmaheap_unify/`](../07_results/dmaheap_unify/summary.md) 5단계에서, **sync를 끄면 틀린 값이
나온다는 것을 보이려고** 시나리오 세 개를 만들었다.

| 변종 | 방법 | 결과 |
|---|---|---|
| ① full-dirty | 전체를 `memset(0xA5)` 하고 `END｜WRITE` 생략 | diff **0** (재현 실패) |
| ② line-dirty | 64 B 캐시 라인마다 1바이트만 써서 부분 dirty 강제 | diff **0** (재현 실패) |
| ③ stale-read | 프레임 B로 캐시를 채운 뒤 DMA가 프레임 A를 쓰고, `START｜READ` 없이 읽음 | diff **0** (재현 실패) |

**셋 다 실패했다.** 그런데 `perf record -g` 로 **이유를 찾았다** — Hailo 드라이버가
장치 쪽에서 스트리밍 DMA API로 캐시 관리를 이미 하고 있었다
(`hailo_vdma_buffer_map → __dma_sync_single_for_device → __pi_dcache_clean_poc`).
디바이스 트리에 `dma-coherent` 가 0건이므로 커널이 비일관 장치로 취급하고 드라이버가 대신해 주는 것이다.

**그런데도 sync를 유지한다.** 그 문서의 논리가 그대로 옮길 만하다:

1. **HailoRT 4.24.0 + `hailo_pci` 구현에 의존하는 관찰**이지 스펙도 계약도 아니다.
   바뀌면 조용히 깨지고 증상은 "간헐적 오검출"이라 추적이 거의 불가능하다.
2. **V3D 방향은 이 증거가 덮지 못한다.** V3D는 import 시점에 한 번 attach하고 프레임마다
   다시 map하지 않으므로 위의 프레임별 캐시 관리가 없다.
   그런데도 GPU 바이트 합이 항상 맞았고 — **기제는 미확정이다.**
3. 앞으로 이 버퍼를 **장치가 아닌 소비자**(녹화 스테이징, 트래킹)가 쓰면 이 보호가 사라진다.

> "**실험이 '필요하다'를 반증한 것이 아니라 '반증하지 못한' 것이다.**"

**이 문장이 이 챕터의 요약이기도 하다.** "테스트를 통과했으니 괜찮다"와
"규약을 지켰으니 괜찮다"는 다르다. 위 2번 — **V3D 방향의 일관성 기제는 여전히 미확정 가설**이다.

---

## 6.11 ⚠ `--postproc-alloc heap` 은 쓰지 마라

후처리 출력 버퍼를 **dma-heap에서** 잡는 선택지(`--postproc-alloc heap`)가 코드에 있다.
장점이 실측으로 확인됐다 ([`dmaheap_unify/summary.md`](../07_results/dmaheap_unify/summary.md)):

- V3DV가 dma-heap fd를 **import 한다** (6/6 성공)
- **GPU 읽기 성능 손해가 없다** (heap/plain = 0.9994)
- **CPU 읽기가 3배 빠르다** (heap 0.262 ms vs V3D plain 0.789 ms, 1.22 MB 기준)
- 정확성 영향 없음 (후보 집합이 바이트 단위 동일)

그런데 **기본값은 `export`(Vulkan 할당)이고, `heap` 은 쓰지 말라고 적혀 있다.** 이유는 둘이다.
① **3자 일관성(PCIe DMA ↔ CPU 캐시 ↔ V3D)이 미검증이다**(§6.10의 V3 실험이 "필요 없다"를
증명하지 못했고 V3D 쪽 기제도 미확정이다). ② **오늘의 이득이 0이다** — 현재 핫패스에
**CPU가 V3D 메모리를 크게 읽는 곳이 없다**(그렇게 만든 것이 이 프로젝트의 성과다, §6.7).
CPU 읽기가 3배 빨라져 봐야 읽는 곳이 없으니 이득이 없다.

> "51.6 FPS / 검출 9457 / P6 2.12 ms / P8 4.23 ms에 정확히 맞춰진 게이트를
> **이득 0인 변경으로 흔드는 것은 나쁜 거래다.**"
> — [`dmaheap_unify/summary.md`](../07_results/dmaheap_unify/summary.md) 결론 3

**그러면 왜 만들었나?** 두 가지를 얻으려고다.
① **설계 제약을 근거로 닫기** — "GPU가 쓴 버퍼를 CPU로 읽지 마라"가 앞으로도 절대 규칙인지,
회피 수단이 있는지를 확인했다. **있다.** 나중에 녹화나 추적 기능을 붙일 때 쓸 카드가 생겼다.
② **메모리 관리 일원화** — 같은 일을 하던 구현 셋이 하나로 모이면서 §6.10의 결함 둘이 사라졌다.

---

## 6.12 ★ V3DV 헤드룸 +256 B 함정

이 절은 "스펙대로 했는데 셰이더가 데이터를 잘라 읽는" 버그의 이야기다.

### 증상

원시 출력 텐서 중 `conv71_178` 은 **409,600 바이트**다. 16384의 정확히 25배라
dma-buf를 409,600 B로 딱 맞게 잡았다. 그런데 셰이더가 마지막 바이트들을 못 읽었다.

로그: `range 409,344 < hw 409,600`

### 원인

**V3DV는 스토리지 버퍼의 메모리 요구 크기에 여유를 더한다.**
TMU(Texture Memory Unit)가 데이터를 **미리 앞질러 읽기** 때문이다.
실측으로 **크기와 무관하게 항상 +256 B** 였다.

```
   요청: 409,600 B 짜리 STORAGE 버퍼를 만들어 줘
   V3DV: vkGetBufferMemoryRequirements2 → req.size = 409,856 B   (+256)
```

그런데 **import 메모리의 크기는 dma-buf 크기(409,600)로 고정이다.** 늘릴 수 없다.
`vkBindBufferMemory` 가 실패하거나, 검증 레이어가 잡는다.

[`dmabuf_import.cpp`](../01_src/vulkan/dmabuf_import.cpp) 는 이때 **VkBuffer 범위를 줄여서**
대응한다 — 버퍼를 409,344 B로 다시 만들면 요구가 409,600이 되어 들어맞는다.
그런데 그러면 **셰이더가 마지막 256 B를 못 본다.**

```
   dma-buf         [■■■■■■■■■■■■■■■■■■■■■■■■■■■■] 409,600 B (실데이터)
   VkBuffer range  [■■■■■■■■■■■■■■■■■■■■■■■■■■··] 409,344 B
                                                ↑ 셰이더가 못 읽는 256 B
```

### 해법 — 할당할 때 미리 더 잡는다

[`vk_storage_headroom.hpp`](../01_src/vulkan/vk_storage_headroom.hpp) 가 하는 일이다.

```
   ① 더미 VkBuffer 를 만든다 (size = 요청 크기, usage = STORAGE,
                              pNext = VkExternalMemoryBufferCreateInfo{DMA_BUF})
   ② vkGetBufferMemoryRequirements2  → req.size
   ③ headroom = req.size - 요청 크기            ← V3D 실측: 항상 256
   ④ 더미를 파괴한다  (할당은 한 번도 하지 않는다)
   ⑤ dma-heap 에서 align_up(요청 + headroom, 16384) 로 잡는다
        → 409,600 + 256 = 409,856 → 16384 배수로 올려 425,984 B
        → VkBuffer range 425,728 ≥ 409,600  ✔
```

### ★ 왜 상수 256을 박지 않는가

헤더 주석이 명시한다: "**상수 256을 박지 않는다** — 드라이버 의존이다.
질의 결과는 크기별로 memoize한다."
Mesa가 업데이트되어 값이 288이 되면 상수를 박은 코드는 **조용히 틀린 데이터를 읽는다.**
질의하는 코드는 그냥 맞는다.

### 과할당이 스펙 위반이 아닌가

요청보다 큰 메모리를 할당하는 것이 문제 되지 않나? **전용 할당을 쓰면 문제가 된다** —
`VUID-VkMemoryDedicatedAllocateInfo-buffer-02965`(validusage.json, Vulkan 1.4.309)가
`allocationSize == req.size` 를 요구하기 때문이다. 그런데 이 조건은
**`VkMemoryDedicatedAllocateInfo` 를 붙였을 때만** 적용되고, 실측에서 V3D는 6/6
`requiresDedicatedAllocation = prefersDedicatedAllocation = 0` 이라
[`dmabuf_import.cpp`](../01_src/vulkan/dmabuf_import.cpp) 가 그 구조체를 **붙이지 않는다**
→ 과할당이 걸리지 않는다. **스펙 원문을 읽고 "이 VUID가 적용되는 조건"까지 확인한 것**이
이 판단의 핵심이고, 헤더 주석에 그 근거가 출처와 함께 적혀 있다.

### 그리고 조용히 넘기지 않는다

`DmaBufImportCache::get(fd, size, min_range)` 에 `min_range` 를 주면 결과 range가 그보다
작을 때 **예외를 던진다**(캐시 적중에서도 검사한다).
코드 주석: "조용히 넘기면 셰이더가 잘못된 데이터를 읽으므로 여기서 끊는다."
**데이터가 조용히 잘리는 것보다 프로그램이 죽는 편이 낫다**는 판단이다.

---

## 6.13 HailoRT의 16384 B 정렬 요구

NPU에게 버퍼를 주려면 **시작 주소와 크기가 16384 B의 배수**여야 한다.
안 맞으면 어떻게 되나? **실패하지 않는다.** 대신 프레임마다, 스트림마다 경고가 뜬다:

```
vdma_stream.cpp:379  unaligned buffer causes performance degradation
```

4096 B 정렬로 첫 실행을 했을 때 이 경고가 **1,477건** 나왔다
([`gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md) "주의와 남은 일").
그래서 [`dmabuf_export.hpp`](../01_src/vulkan/dmabuf_export.hpp) 의 기본 정렬
(`kHailoAsyncAlign`)이 16384다.

### 두 가지를 모두 맞춘다

**① 할당 크기**: `align_up(VkMemoryRequirements::size, align)`.
여기서 §6.12의 전용 할당 제약이 다시 나온다 — 전용 할당을 쓰면 크기를 올릴 수 없으므로
**드라이버가 요구할 때만** 쓴다. V3D는 요구하지 않는다(실측).

**② 시작 주소**: dma-buf 경로는 오프셋 0이라 자동으로 맞는다(§6.5 ③).
호스트 포인터 경로는 `vkMapMemory` 가 돌려주는 주소가 mmap 정렬(보통 페이지 크기)이라 보장이 없다.
안 맞으면 **align만큼 여유를 더해 한 번 다시 할당하고** 정렬된 지점을 노출한다.

> 실제로는 매핑이 항상 16 KB 정렬로 나와서 **이 재할당 경로는 한 번도 실행되지 않았다**
> ([`display_vulkan_hdmi/summary.md`](../07_results/display_vulkan_hdmi/summary.md) "주의와 남은 일":
> "`DmaBufExporter`의 '정렬 미스 시 재할당' 경로는 실행되지 않았다 → 미테스트 경로").
> **테스트되지 않은 코드라는 사실을 문서에 적어 둔 것**이 여기서의 요점이다.

### 크기가 세 가지인 이유

[`dmabuf_export.hpp`](../01_src/vulkan/dmabuf_export.hpp) 가 크기를 세 개 노출한다.

| 이름 | 뜻 | 예 |
|---|---|---:|
| `requested_size` | 호출자가 요청한 바이트 수 (텐서 실크기) | 1,228,800 |
| `alloc_size` | `vkAllocateMemory` 에 넘긴 크기 (align 배수) | 1,245,184 |
| `dmabuf_size` | `lseek(fd, 0, SEEK_END)` 로 실측한 dma-buf 객체 크기 | 1,245,184 |

왜 셋이 필요한가? **HailoRT API마다 받는 크기가 다르기 때문이다.**

- `dma_map_dmabuf(fd, size, D2H)` — dma-buf 전체 크기를 준다(버퍼 전체를 덮는 쪽이 안전)
- `read_async` / `hailo_dma_buffer_t.size` — **`hw_frame_size` 만 받는다**

그리고 여기에 **치명적인 함정**이 있다:

> "⚠ **`infer_async` 에 틀린 크기를 넘기면 `HAILO_SUCCESS` 를 돌려주고 콜백을 하나도 호출하지 않는다.**
>  요청이 내부 워커에서 조용히 버려져 **무한 대기하는 코드는 영구 정지한다**"
> — [`gpu_postprocess/summary.md`](../07_results/gpu_postprocess/summary.md)

그래서 러너는 프레임 완료 대기에 **반드시 시한(timeout)** 을 두고, 시한 초과를 치명적
오류로 올려 종료한다. (HailoRT 쪽 상세는 [04_npu_hailort.md](04_npu_hailort.md).)

---

## 6.14 자가 점검

**Q1. dma-buf를 쓰면 자동으로 zero-copy가 되는가?**

**아니다.** `--mode dmabuf --layout nhwc` 는 CPU 비용이 **17.6 ms/frame** 으로
`--mode ptr --layout nhwc` 와 **똑같다**. HailoRT가 그 버퍼를 CPU로 읽어 전치하기 때문이다.
zero-copy는 **소비자가 그 메모리를 CPU로 건드리지 않을 때** 성립한다.
이 프로젝트에서는 `dmabuf + nhcw` 조합만 성립하고, 그때 CPU가 **3.5 ms** 로 떨어진다.
근거: [`REPORT.md §4.3`](../07_results/pipeline_ffmpeg_vulkan_hailo/REPORT.md) 근거 2·3.

**Q2. GPU가 쓴 버퍼를 CPU가 읽는 것이 왜 그렇게 비싼가? 반대 방향은?**

V3D가 노출하는 Vulkan 메모리 타입이 `DEVICE_LOCAL｜HOST_VISIBLE｜HOST_COHERENT` **하나뿐**이고
**`HOST_CACHED` 가 없다.** 즉 CPU 쪽에서 캐시되지 않는 매핑이다.
캐시가 없으면 **읽기는 매번 메모리까지 가서 기다려야** 한다 —
같은 1.22 MB를 순차로 읽으면 1.479 GB/s인데 stride-3으로 읽으면 약 **0.088 GB/s**,
16배 차이다([`pipeline_ffmpeg_vulkan_hailo/summary.md`](../07_results/pipeline_ffmpeg_vulkan_hailo/summary.md)).
**반대 방향(CPU→GPU 쓰기)은 싸다** — 쓰기는 버퍼에 던지고 바로 다음 일을 하기 때문이다.

**Q3. `dmabuf_import.cpp` 는 왜 `dup(fd)` 를 해서 넘기는가?**

Vulkan 스펙(1.4.309, File Descriptor External Memory)이 이렇게 말하기 때문이다:
"Importing memory from a file descriptor **transfers ownership of the file descriptor**
from the application to the Vulkan implementation. The application must not perform any
operations on the file descriptor after a successful import."
즉 성공하면 그 fd는 더 이상 우리 것이 아니다. 원본을 계속 쓰려면 사본을 넘겨야 한다.
`vkFreeMemory` 를 하면 그 사본이 함께 닫힌다. **실패했을 때는 넘어가지 않았으므로 우리가 `close` 한다.**

**Q4. V3DV의 "+256 B"는 무엇이고, 왜 코드에 256을 박지 않는가?**

V3DV는 스토리지 버퍼의 메모리 요구 크기에 TMU 선읽기용 여유를 더한다(V3D 실측: 크기와 무관하게 +256 B).
dma-buf를 데이터 크기에 **딱 맞게** 잡으면 요구가 초과되므로 `DmaBufImportCache` 가
VkBuffer 범위를 줄이고 → **셰이더가 마지막 바이트를 못 읽는다**(실제 사례: `range 409,344 < hw 409,600`).
해법은 할당 시점에 미리 헤드룸을 더하는 것이고, 그 값은 **더미 버퍼 하나를 만들어
`vkGetBufferMemoryRequirements2` 로 질의**해서 얻는다([`vk_storage_headroom.hpp`](../01_src/vulkan/vk_storage_headroom.hpp)).
**256을 박지 않는 이유는 드라이버 의존 값이기 때문이다** — 드라이버가 바뀌면 상수 코드는
조용히 틀린 데이터를 읽지만 질의 코드는 그냥 맞는다.

**Q5. `DMA_BUF_IOCTL_SYNC` 가 필요하다는 것을 실험으로 증명하지 못했는데 왜 유지하는가?**

세 변종(full-dirty / line-dirty / stale-read) 전부 실패를 재현하지 못했고,
`perf` 로 **Hailo 드라이버가 장치 쪽에서 이미 캐시 관리를 하고 있다**는 것까지 확인했다.
그런데도 유지하는 이유는 셋이다 — ① 그것은 HailoRT 4.24.0 + `hailo_pci` 구현에 의존하는
**관찰이지 스펙이 아니다**(바뀌면 조용히 깨지고 증상이 "간헐적 오검출"이라 추적 불가),
② **V3D 방향은 이 증거가 덮지 못한다**(기제 미확정 — 왜 항상 맞는지 설명하지 못했다),
③ 나중에 장치가 아닌 소비자가 이 버퍼를 쓰면 그 보호가 사라진다.
**"실험이 '필요하다'를 반증한 것이 아니라 반증하지 못한 것이다."**
근거: [`dmaheap_unify/summary.md`](../07_results/dmaheap_unify/summary.md) 5단계 V3.

**Q6. `--postproc-alloc heap` 은 CPU 읽기가 3배 빠른데 왜 기본값이 아닌가?**

① **3자 일관성(PCIe DMA ↔ CPU 캐시 ↔ V3D)이 미검증**이고, ② **오늘의 이득이 0**이기 때문이다.
현재 핫패스에는 CPU가 V3D 메모리를 크게 읽는 곳이 아예 없다 — CPU가 읽는 것은 GPU가 압축한
후보 약 1.6 KB뿐(1.1~4.0 µs)이다. CPU 읽기가 3배 빨라져도 읽는 곳이 없으면 이득이 없고,
정밀하게 맞춰진 게이트만 흔들린다. 기본값은 `export` 다.

---

**다음** → [07_video_display.md](07_video_display.md) (이 dma-buf 안에 실제로 들어 있는 것이
무엇인가 — SAND128 NV12라는 형식과, 그것을 화면에 내보내는 법)
