# dma-heap 일원화 — 1단계 P0 관문 (`--probe heapimport`)

계획서(`~/.claude/plans/jaunty-dazzling-quasar.md`) **부록 B.13 1단계**. 판정 기준은 **B.14 S1**.

> **한 줄 결론: V3DV 는 dma-heap fd 를 import 한다. 관문은 열렸다.**
> 단 **6텐서 전부 `range = size − 256`** 이고, `conv71_178` 은 그 256 B 가 실데이터를 잘라먹는다
> → **조건부 합격**. B.12 헤드룸이 필수다.

---

## 1. 측정 조건

| 항목 | 값 |
|---|---|
| 실행 | `build/01_src/vulkan/yvo_postproc_probe --probe heapimport --validation --sync-validation` |
| 일시 / 온도 | 2026-09-20 16:18 KST / `measure_temp=65.9 °C`, `get_throttled=0xe0000`(과거 비트만) |
| **NPU** | **쓰지 않는다.** HailoRT 를 한 번도 부르지 않는다 |
| GPU | `V3D 7.1.10.2` / `DRIVER_ID_MESA_V3DV` / V3DV Mesa **26.2.2** / apiVersion 1.3.354 |
| 메모리 타입 | **정확히 1개** — type 0 = `DEVICE_LOCAL｜HOST_VISIBLE｜HOST_COHERENT` (0x007) |
| dma-heap | `/dev/dma_heap/vidbuf_cached` (**`system` 으로의 심볼릭 링크**, 실측) |
| 할당 크기 | `align_up(hw_frame_size, 16384)`, `fd_flags = O_RDWR｜O_CLOEXEC` |
| import 경로 | **기존 `01_src/vulkan/dmabuf_import.cpp` 의 `DmaBufImportCache::get(fd, dmabuf_size)` 를 그대로 호출.** 새 import 코드를 쓰지 않았다 |
| 왕복 셰이더 | `02_shaders/heap_sum.comp`(신규). CPU 가 `mmap(heap_fd)` 로 쓴 패턴(`word[i] = i × 2654435761`)의 바이트 합을 GPU 가 재계산 |
| 증거 | `evidence/p0_heapimport.txt`, ASan 실행 `verification/asan_p0_heapimport.txt`, 환경 `env_bench.txt` |

**주의**: hw_frame_size 6개는 `04_data/raw_tensors/fixtures.json` 에서 읽는다. 상수를 박지 않았다.

---

## 2. 결과 — 텐서별 전수

| 텐서 | hw (B) | 요청 = align_up(hw,16384) | dma-buf 실크기 (lseek) | **① `memoryTypeBits`** | **② req.size (size=dmabuf)** | 여유 | **④ `range`** | 축소 | `range ≥ hw` | **⑦ 왕복** |
|---|---:|---:|---:|:---:|---:|---:|---:|---:|:---:|:---:|
| `conv102_178` | 30,720 | 32,768 | 32,768 | **0x00000001** | 33,024 | +256 | 32,512 | −256 | 예 | 일치 |
| `conv105_178` | 38,400 | 49,152 | 49,152 | **0x00000001** | 49,408 | +256 | 48,896 | −256 | 예 | 일치 |
| **`conv71_178`** | **409,600** | **409,600** | **409,600** | **0x00000001** | 409,856 | +256 | **409,344** | −256 | **아니오 (−256)** | 일치 |
| `conv74_178` | 512,000 | 524,288 | 524,288 | **0x00000001** | 524,544 | +256 | 524,032 | −256 | 예 | 일치 |
| `conv87_178` | 102,400 | 114,688 | 114,688 | **0x00000001** | 114,944 | +256 | 114,432 | −256 | 예 | 일치 |
| `conv90_178` | 128,000 | 131,072 | 131,072 | **0x00000001** | 131,328 | +256 | 130,816 | −256 | 예 | 일치 |

모든 텐서 공통:

| 항목 | 값 |
|---|---|
| **③ `requiresDedicatedAllocation`** | **`VK_FALSE`** (6/6) |
| **③ `prefersDedicatedAllocation`** | **`VK_FALSE`** (6/6) |
| `VkExternalBufferProperties` 기반 `dmabuf_import_dedicated_only()` | `false` |
| `VkMemoryRequirements2::alignment` | **256 B** |
| `VkMemoryRequirements2::memoryTypeBits` | `0x00000001` |
| **⑤ `vkAllocateMemory`** | **`VK_SUCCESS`** (6/6). `dmesg` 수집 불필요 |
| **⑥ `mmap` 포인터 16384 정렬** | **6/6 정렬됨** (HailoRT 요구, 부록 A.4 만족) |
| `DMA_BUF_IOCTL_SYNC` `START｜WRITE` / `END｜WRITE` | 반환 0 (6/6) |
| 검증 레이어 errors (`--validation --sync-validation`) | **0** |
| fd 누수 | **0** (heap/import 스코프 진입 전 14 → 종료 후 14) |

### B.12 가 캐시할 **헤드룸** 값 (직접 측정)

`hw_frame_size` 그대로의 크기로 더미 `VkBuffer`(`usage=STORAGE_BUFFER`,
`pNext=VkExternalMemoryBufferCreateInfo{DMA_BUF}`)를 만들고 `vkGetBufferMemoryRequirements2` 로 물은 값:

| 텐서 | `req(size = hw)` | **headroom** |
|---|---:|---:|
| 6텐서 전부 | `hw + 256` | **+256 B (상수)** |

**V3DV 의 스토리지 버퍼 여유는 크기와 무관한 상수 +256 B 이고 `alignment` 는 256 B 다.**
B.12 의 `DmaHeapConfig::device_headroom` 은 **하드코딩 대신 이 질의로 채우면 된다** — 절차가 실측으로 확인됐다.

---

## 3. 결론

### 3.1 S1 판정: **조건부 합격**

| B.14 S1 의 갈래 | 실측 |
|---|---|
| `memoryTypeBits == 0` / `vkGetMemoryFdPropertiesKHR` 실패 → 하드 스톱 | **해당 없음.** 6/6 이 `0x1` |
| `vkAllocateMemory` 실패 → 불합격 | **해당 없음.** 6/6 `VK_SUCCESS` |
| 왕복 불일치 → 일관성 문제 | **해당 없음.** 6/6 일치 |
| `range == size` → 합격 | **해당 없음** |
| **`range < size` → 조건부 합격, B.12 헤드룸 필요** | **← 이것이다. 6/6 축소** |

**부록 B.4 의 미지수 (a) 가 닫혔다**: *"V3DV 가 `/dev/dma_heap/system` fd 를 import 하는가"* → **한다.**
그리고 **형식적 성공에 그치지 않는다** — CPU 가 `mmap(heap_fd)` 로 쓴 패턴을 GPU 셰이더가
바이트 단위로 같은 합으로 읽었다(6/6). **이중 매핑(Vulkan import + heap fd mmap)이 실제로 성립한다.**

### 3.2 B.6(a) 함정은 **예측대로 실물이다**

`conv71_178`(80×80×64 = 409,600 = 16384×25)은 `align_up(hw, 16384)` 여유가 정확히 0이라
**셰이더가 볼 수 있는 범위가 409,344 B 로 줄어 마지막 256 B 를 읽지 못한다.**
NHCW 인덱스 `((h·C)+ch)·W + w` (C=64, W=80) 기준으로 잃는 구간은 계획서가 적은 대로
**h=79(P3 격자 마지막 행)의 coord=3, bin=12~15** 다.

이 프로브의 왕복이 "일치"로 나온 것은 **CPU 도 GPU 와 같은 `range` 범위만 더했기 때문**이다.
**왕복 일치가 이 손실을 면제해 주지 않는다.** 두 사실은 독립이다.

### 3.3 헤드룸 해법은 spec 위반이 아니다 (③ 이 확인해 준 것)

`prefersDedicatedAllocation == VK_FALSE` 이므로 **`VkMemoryDedicatedAllocateInfo` 를 붙이지 않아도 되고**,
따라서 `allocationSize` 를 `VkMemoryRequirements::size` 보다 크게(= dma-buf 실크기로) 잡는 과할당이
`VUID-VkMemoryDedicatedAllocateInfo-buffer-02965` 에 걸리지 않는다. **부록 A.11 의 벽이 import 경로에는 없다.**

B.12 절차를 이 수치에 대입하면(**계산이며 아직 측정한 것이 아니다 — 3단계에서 확인할 것**):

```
conv71_178: requested = 409,600
  req.size(409,600) = 409,856                                   ← 측정값
  heap 할당 = align_up(max(409600, 409856), 16384) = 425,984    (16384 × 26)
  import 시 VkBuffer 축소 → range = 425,984 − 256 = 425,728  ≥ 409,600  OK
  비용: 이 텐서만 +16,384 B
```

### 3.4 다음 단계에 넘기는 값

- `device_headroom = 256 B`(상수, 크기 무관), `alignment = 256 B` — **질의로 채우고 상수로 박지 마라**
- import 메모리 타입은 항상 type 0 하나다. `memoryTypeBits` 교집합 계산은 형식적이지만 그대로 두는 게 맞다
- **GPU 접근 성능(B.2 의 주 미지수)은 이 프로브가 재지 않았다.** 4단계 A/B 측정이 그대로 남아 있다

---

## 4. 주의 — 이 프로브가 **말하지 않는 것**

1. **성능을 재지 않았다.** 캐시된 dma-heap 버퍼를 GPU 가 읽을 때 V3D 자기 메모리만큼 빠른지는
   여전히 미측정이다(B.2). 본문의 수치는 전부 크기·플래그·정확성이다.
2. **NPU 가 개입하지 않은 2자 경로다.** 확인된 일관성은 *CPU 캐시 매핑 쓰기 → V3D 읽기* 뿐이다.
   **B.10 의 3자 일관성(PCIe DMA ↔ CPU 캐시 ↔ V3D)은 검증되지 않았다** — V1·V2·V3(5단계)가 그 몫이다.
3. **sync ioctl 이 필요하다는 증거도, 불필요하다는 증거도 아니다.** 이 프로브는 항상 `START/END|WRITE` 를
   걸고 돌렸다. 빼고 돌려 보지 않았다(부록 A P3 ④ 교훈: 안 틀려도 불필요의 증거가 아니다).
4. **읽기 방향(`START/END|READ`) sync 는 한 번도 쓰지 않았다.** GPU→CPU 방향은 미검증이다.
5. **`range` 축소는 V3DV 26.2.2 의 관측값이다.** 드라이버가 바뀌면 +256 이 달라질 수 있다.
   그래서 B.12 가 상수를 박지 말라고 한 것이고, 이 프로브가 그 질의 절차를 실증했다.
6. `vidbuf_cached` 와 `system` 은 **같은 힙**이다(심볼릭 링크). 둘을 비교 측정한 것이 아니다.
7. 1회 실행이다. `measure` 스킬의 3회 인터리브·중앙값 규약은 성능 측정(4단계)에 적용된다.

---

## 5. 재현

```bash
cmake -S . -B build -G Ninja && cmake --build build
build/01_src/vulkan/yvo_postproc_probe --probe heapimport --validation --sync-validation
# → 07_results/dmaheap_unify/evidence/p0_heapimport.txt
# 종료 코드 0 = PASS (하드 스톱·왕복 불일치·검증레이어 error·fd 누수면 1)
```

`--heap PATH` 로 힙 노드를 바꿀 수 있다(기본 `vidbuf_cached` → `system` 폴백).

## 6. 참조한 스펙 근거

로컬 레지스트리 **Vulkan 1.4.309**(commit 43d153dd, 2025-02-21). `mcp__vulkan-docs__*` 는 이 세션에 로드되지 않아 쓰지 않았다.

| 출처 | 확인한 것 |
|---|---|
| `validusage.json` `VUID-VkMemoryDedicatedAllocateInfo-buffer-02965` | "buffer 가 VK_NULL_HANDLE 이 아니고 … `VkMemoryAllocateInfo::allocationSize` 는 그 버퍼의 `VkMemoryRequirements::size` 와 같아야 한다" → prefers/requires 가 false 라 이 제약을 피할 수 있음을 확인 |
| `validusage.json` `VUID-VkImportMemoryFdInfoKHR-handleType-00669` / `-00667` | handleType 은 OPAQUE_FD 또는 **DMA_BUF_BIT_EXT** 여야 하고, import 지원이 `VkExternalBufferProperties` 로 보고돼야 한다 |
| `validusage.json` `VUID-VkBufferCreateInfo-pNext-00920` | `VkExternalMemoryBufferCreateInfo::handleTypes` 는 `compatibleHandleTypes` 안이어야 한다 |
| `validusage.json` `VUID-vkGetMemoryFdPropertiesKHR-handleType-00674` | handleType 이 OPAQUE_FD 면 안 된다(DMA_BUF 는 적법) |
| `vk.xml` | `VkMemoryFdPropertiesKHR{sType,pNext,memoryTypeBits}`, `VkMemoryDedicatedRequirements{prefers,requires}`, `VkImportMemoryFdInfoKHR{handleType,fd}`, `VkExternalMemoryBufferCreateInfo{handleTypes}` 멤버 확인. `VK_EXT_external_memory_dma_buf` 는 `VK_KHR_external_memory_fd` 에 의존 |
