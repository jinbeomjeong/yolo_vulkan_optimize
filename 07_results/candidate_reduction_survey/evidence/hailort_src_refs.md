# HailoRT 소스 근거 — inter-context·DDR·config 버퍼가 PCIe를 오가는가

- 대상: `hailo-ai/hailort` 태그 **v4.24.0** (커밋 `63adffec12da88f1e7cfdc07f2e07681671e83e8`, 2026-06-17).
  설치된 런타임(`libhailort.so.4.24.0`, FW 4.24.0)과 같은 버전이다.
- 읽은 방법: `git clone --depth 1 --branch v4.24.0 https://github.com/hailo-ai/hailort.git` (세션 스크래치패드, 저장소 밖).
- URL 형식: `https://github.com/hailo-ai/hailort/blob/v4.24.0/<경로>#L<줄>`. 아래 경로는 저장소 루트 기준이다.
- "확인"은 소스나 측정으로 직접 확인한 것이고, "추정"은 해석이다.

## 1. 중간 버퍼(inter-context·DDR)는 호스트 메모리에 있다 — 확인

| 근거 | 경로:줄 | 내용 |
|---|---|---|
| 중간 버퍼 생성 | `hailort/libhailort/src/core_op/resource_manager/internal_buffer_manager.cpp:39-69` | `create_intermediate_sg_buffer` → `vdma::SgBuffer::create(driver, size, DmaDirection::BOTH)`, CONTINUOUS면 `ContinuousBuffer`(CMA). 둘 다 호스트 메모리다 |
| SG 버퍼 = 호스트 할당 | `hailort/libhailort/src/vdma/memory/sg_buffer.cpp:16-18` | `MappedBuffer::create_shared_by_allocation(size, driver, dir)` |
| 할당 방식 | `hailort/libhailort/src/vdma/memory/dma_able_buffer.cpp:63-89`, `hailort/common/os/posix/mmap_buffer.cpp:33-36` | Linux에서 `mmap(MAP_ANONYMOUS \| MAP_SHARED)` 페이지를 드라이버가 DMA 매핑 |
| DDR·IC 모두 같은 관리자 | `hailort/libhailort/src/core_op/resource_manager/resource_manager.cpp:505-530` | `create_intermediate_edge_layer`: `layer_type`이 `DDR` 또는 `INTER_CONTEXT`일 때만 허용, 버퍼는 `m_internal_buffer_manager->get_intermediate_buffer()`에서 온다. DDR은 circular |
| 실측 로그 | `evidence/buffer_alloc_b{1,8}.txt` (`internal_buffer_manager.cpp:77`의 info 줄) | b1 `CMA=0 Pinned=7,835,648`, b8 `CMA=0 Pinned=47,976,448` — **중간 버퍼 전부가 pinned 호스트 RAM** |

따라서 Hailo-8(PCIe)에서 컨텍스트 k가 내보낸 inter-context 텐서는 PCIe D2H로 호스트 RAM에 쓰이고,
컨텍스트 k+1(또는 그 뒤)이 PCIe H2D로 다시 읽는다. "DDR" 채널도 이름과 달리 칩 외부 DRAM이 아니라
호스트 RAM의 circular 버퍼다(CMA=0, Pinned에 포함). `add_ddr_pair_info`(fw-actions)가 D2H 채널 22 ↔ H2D 채널 3을
짝지어 같은 컨텍스트 안에서 쓰고 곧바로 읽는다.

## 2. 버퍼 크기 규칙 — 확인

| 근거 | 경로:줄 | 내용 |
|---|---|---|
| inter-context | `.../resource_manager/internal_buffer_planner.cpp:316-350` | 키 = (생산 컨텍스트, 출력 스트림). 크기 = `get_layer_transfer_size(layer_info)` × `batch_size`. 소비 컨텍스트가 여럿이면 `end_context`만 늘린다(버퍼 1개, 읽기 여러 번) |
| DDR | `.../resource_manager/internal_buffer_planner.cpp:293-314` | 크기 = `core_bytes_per_buffer`(행) × `min_buffered_rows`. 배치와 무관한 circular 버퍼 |
| 로그의 factor | `.../resource_manager/internal_buffer_planner.cpp:421-446` | `factor = (cma + pinned) / Σ edge_layer buffer_size` → `Σ = pinned / factor` |
| IC 디스크립터 | `.../resource_manager/resource_manager_builder.cpp:200-205`, `resource_manager.cpp:520-526` | IC는 `frame_credits = get_layer_transfer_size()` 단위로 `batch_size` 회 프로그램 |

b1/b8 두 점에서 나눈 결과(`buffer_alloc_b8.txt`): 프레임당 IC 버퍼 6,656,000 B(= HEF의 비패딩 IC 출력 합과 정확히 일치),
배치 무관 DDR 2,102,784 B.

## 3. config(가중치 포함 CCW)도 매 배치 호스트에서 PCIe로 가져온다 — 확인

| 근거 | 경로:줄 | 내용 |
|---|---|---|
| CCW 섹션 매핑 | `.../resource_manager/resource_manager.cpp:1074-1097` | HEF 파일 안의 CCW 섹션(호스트 메모리)을 `DmaDirection::H2D`로 매핑("aligned ccws") |
| config 디스크립터 | `.../resource_manager/config_buffer.cpp:22-33, 98-156` | CCW DMA 전송마다 디스크립터를 프로그램, 앞에 NOP 패딩 1개 |
| 페이지 크기 | `.../resource_manager/config_buffer.cpp:299-311`(`FORCE_DEFAULT_PAGE_SIZE = true`), `hailort/libhailort/src/vdma/memory/descriptor_list.hpp:42`(`DEFAULT_SG_PAGE_SIZE = 512`), `vdma/memory/buffer_requirements.cpp:181-200` | config 디스크립터 페이지는 512 B. 디스크립터 수가 한도를 넘으면 키우고 경고를 남기는데, 로그에 그 경고는 없다 |
| fetch 액션 | `.../resource_manager/resource_manager_builder.cpp:806-826` | 연속된 WriteDataCcw 묶음 끝마다 `FetchCfgChannelDescriptorsAction` 추가 → fw-actions의 `fetch_cfg_channel_descriptors{descriptors_count}` |
| 선행 fetch 없음 | `.../resource_manager/resource_manager_builder.cpp:993` | `support_pre_fetch = is_hailo1x_device_type(...)` → **Hailo-8은 다음 컨텍스트 config를 미리 가져오지 않는다**(컨텍스트 안에서 직렬) |
| HEF 필드 | `hailort/libhailort/hef.proto:600-609` | `ProtoHEFActionWriteDataCcwPtr { offset, size, cfg_channel_index }` — 컨텍스트별 config 바이트의 원본 |

검산(`fwactions_descriptor_sums.txt`): 컨텍스트별 `descs × 512` ÷ HEF `Σ ccw_ptr.size` = 1.0012~1.0034.
전체(preliminary + dynamic) 56,347 × 512 = 28,849,664 B ↔ HEF 헤더의 CCW 섹션 28,802,496 B(+0.16%).
b1과 b8의 디스크립터 수가 같으므로 **config는 배치당 1회**다(배치 크기와 무관).

## 4. "디스크립터 수 × 페이지 = 전송 바이트"로 읽어도 되는가

- **config 채널: 상한으로는 맞다(확인).** 전송마다 마지막 디스크립터가 부분 페이지이고 NOP 1개가 붙으므로
  `descs × 512`는 실제보다 0.1~0.3% 크다. 정확한 값은 HEF의 `ccw_ptr.size` 합이다.
- **inter-context·DDR·boundary: fw-actions에서는 계산할 수 없다.** 해당 액션(`activate_inter_context_*`,
  `activate_ddr_buffer_*`)은 채널·스트림 인덱스만 담고 바이트 수가 없다. 그래서 HEF의 edge layer 정보로 계산했다.
- **HailoRT 자체 회계와 일치(확인).** `core_op_metadata.cpp:164-198`의 `get_context_transfer_size()`는
  config burst + boundary 입출력 + DDR 입출력 + IC 입출력을 모두 "전송 바이트"로 더한다(IC·DDR을 PCIe 전송으로 본다는
  HailoRT 쪽 정의). 크기는 `layer_info.hpp:192-205`의 `periph_bytes_per_buffer × periph_buffers_per_frame`.
  이 값을 쓰는 숨은 명령 `hailortcli measure-nnc-performance`(`hailortcli/hailortcli.cpp:202`, HIDDEN;
  `hailortcli/measure_nnc_performance_command.cpp:25-113`; 집계 `resource_manager.cpp:992-1011`)가
  b1에서 304,564 KB / 8프레임을 출력했고, HEF 디코드 합 38,984,288 B/프레임과 KB 절사 범위 안에서 일치한다
  (`nnc_estimator_b1.txt`).
- **남은 모호함(추정).** `conv49`·`concat20` 두 IC 텐서는 코어 행이 폭 24로 패딩돼
  (`core_bytes_per_buffer` 12288/24576) `get_context_transfer_size`는 패딩 포함 값을, IC 디스크립터 프로그래밍과
  버퍼 크기는 비패딩 값(`get_layer_transfer_size`, HEF `padded_width = 20`)을 쓴다. 실제 DMA 바이트가 어느 쪽인지는
  확인하지 못했다. 차이는 프레임당 286,720 B(edge 바이트의 1.6%)로 결론에는 영향이 없다.

## 5. 로그 수준 — 확인

| 근거 | 경로:줄 | 내용 |
|---|---|---|
| 파일 로그 수준 고정 | `hailort/libhailort/src/utils/hailort_logger.hpp:34-36` | release(`NDEBUG`) 빌드는 파일 sink가 `info`로 고정 |
| 환경 변수 | `hailort/common/env_vars.hpp:17-19`, `hailort_logger.hpp:43-51` | `HAILORT_LOGGER_PATH`(파일 위치), `HAILORT_CONSOLE_LOGGER_LEVEL`(콘솔만) |
| 허용 값 | `hailort/libhailort/src/utils/hailort_logger.cpp:228-240` | `info`, `warning`, `error`, `critical`만 허용. `debug`·`trace`는 `HAILO_INVALID_ARGUMENT` |
| 실측 | `evidence/console_run_b1.txt` 2번째 줄 | `Failed to parse console logger level from environment variable: debug` |

결론: 4.24.0 release 라이브러리로는 debug/trace 할당 로그를 얻을 수 없다. 대신 info 수준의 planner 요약 줄
(`internal_buffer_manager.cpp:77`)과 HEF 직접 디코드, fw-actions, 숨은 HW-infer estimator를 교차해 바이트를 구했다.

## 6. HEF 형식 — 확인

| 근거 | 경로:줄 | 내용 |
|---|---|---|
| 헤더 | `hailort/libhailort/src/hef/hef_internal.hpp:115-150` | v2 헤더: magic, version, proto_size, xxh3, `ccws_size` (big-endian) |
| core op 위치 | `hailort/libhailort/src/hef/hef.cpp:925-962` | net-flow HEF는 `network_groups[].ops[].core_op.contexts[]`에 컨텍스트가 있다 |
| IC·DDR hw_shape | `hailort/libhailort/src/hef/hef.cpp:2093-2101, 3340, 3400` | 중간 레이어는 `CORE_HW_PADDING_NOT_SUPPORTED` → padded H/W/F 사용 |
| 연결 종류 | `hailort/libhailort/hef.proto:912-917` | `BOUNDARY=0, INTERMEDIATE=1, DDR=2, CACHE=3` |

이 HEF: version 2, proto 4,865,122 B, CCW 섹션 28,802,496 B(파일 33,667,662 B = 44 + 4,865,122 + 28,802,496).
