# 중복 PNG 대조표

2026-09-21 에 `07_results/` 의 **바이트 일치** PNG 를 정본만 남기고 지웠다(7 MB대 감소).

## 왜 표를 남기나

몇몇 실험은 *두 경로의 산출물이 바이트까지 같다*는 것 **자체가 결론**이다 —
host 경로 == GPU 경로, ISP A안 == C안, batch 4 == batch 8, NHWC == NHCW.
파일이 사라져도 그 사실은 증명 가능해야 하므로 md5 를 남긴다.

남은 파일의 md5 가 아래와 같으면, 지워진 파일도 같은 내용이었다는 뜻이다.

```bash
md5sum 07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame00000.png
```

## 대조표

| md5 | 남긴 파일 | 지운 파일 (내용 동일) |
|---|---|---|
| `8d80e7e6ab0c8d493f0a7d369bc0ccd3` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame00000.png` | `07_results/gpu_postproc_e2e/dump_host/frame00000.png`<br>`07_results/gpu_postproc_e2e/dump_gpu/frame00000.png`<br>`07_results/isp_resize/detections/gpu_nhcw_b8/frame00000.png` |
| `da9f29853dd6f30ccc33f19a60d27f72` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame00300.png` | `07_results/isp_resize/detections/gpu_nhcw_b8/frame00300.png` |
| `b165764b61893b1382676e1445fc67a3` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame00600.png` | `07_results/isp_resize/detections/gpu_nhcw_b8/frame00600.png` |
| `10347c333854e7141c0056b8a08a2a9b` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame00900.png` | `07_results/isp_resize/detections/gpu_nhcw_b8/frame00900.png` |
| `5831e43bb5d430cf13e0f8c6578cee38` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame01200.png` | `07_results/isp_resize/detections/gpu_nhcw_b8/frame01200.png` |
| `a835f4b02891a54b914bb02c475eb387` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame01500.png` | `07_results/isp_resize/detections/gpu_nhcw_b8/frame01500.png` |
| `decf115347901bdc5716b2f388628e5f` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/frame01800.png` | `07_results/isp_resize/detections/gpu_nhcw_b8/frame01800.png` |
| `fba498d1ecafab5545018a3f4ee9fea3` | `07_results/pipeline_ffmpeg_vulkan_hailo/preproc/letterbox_frame100.png`<br>`07_results/isp_resize/samples/letterbox_frame100_nhcw.png` | `07_results/pipeline_ffmpeg_vulkan_hailo/preproc/letterbox_frame100_nhcw.png` |
| `cde8768ac83510fa2104fdfadc8f7ff6` | `07_results/pipeline_ffmpeg_vulkan_hailo/selftest/frame0_detections.png` | `07_results/pipeline_ffmpeg_vulkan_hailo/selftest/b4/frame0_detections.png`<br>`07_results/pipeline_ffmpeg_vulkan_hailo/selftest/b8/frame0_detections.png` |
| `e5f1589658e3c9dc23beb50b8fd2092a` | `07_results/pipeline_ffmpeg_vulkan_hailo/selftest/frame0_letterbox_input.png` | `07_results/pipeline_ffmpeg_vulkan_hailo/selftest/b4/frame0_letterbox_input.png`<br>`07_results/pipeline_ffmpeg_vulkan_hailo/selftest/b8/frame0_letterbox_input.png` |
| `c5bf0ce477fe778efa358b10b1bc1768` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame00000.png` | `07_results/isp_resize/detections/isp_A_filter_lanczos3/frame00000.png`<br>`07_results/isp_resize/detections/isp_C_nhcw_b8/frame00000.png` |
| `c376c66d588729d279da9024686f2092` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame00300.png` | `07_results/isp_resize/detections/isp_C_nhcw_b8/frame00300.png` |
| `7d8a2f1a46e3419c10081ea828d43375` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame00600.png` | `07_results/isp_resize/detections/isp_C_nhcw_b8/frame00600.png` |
| `d2d1861ed526cbfa05eebd46afd90209` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame00900.png` | `07_results/isp_resize/detections/isp_C_nhcw_b8/frame00900.png` |
| `a31486f5f9d5be3d891211757987ae72` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame01200.png` | `07_results/isp_resize/detections/isp_C_nhcw_b8/frame01200.png` |
| `cc6e0b943471aa941b954644ad4aa0a1` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame01500.png` | `07_results/isp_resize/detections/isp_C_nhcw_b8/frame01500.png` |
| `ce33cad429361a2d6bd6aca5ab8ea794` | `07_results/isp_resize/detections/isp_A_nhwc_b8/frame01800.png` | `07_results/isp_resize/detections/isp_C_nhcw_b8/frame01800.png` |

## detections.jsonl 대조표

2026-09-21 에 바이트 일치 `detections.jsonl` 4개(3.0 MB)도 같은 방식으로 정본만 남겼다.
정본 1부는 반드시 남아 있으므로 수치의 1차 근거는 잃지 않는다.

| md5 | 남긴 파일 | 지운 파일 (내용 동일) |
|---|---|---|
| `867a9d9636c614cddd60cf1fe769d0a4` | `07_results/pipeline_ffmpeg_vulkan_hailo/samples/detections.jsonl` | `07_results/gpu_postproc_e2e/dump_host/detections.jsonl`<br>`07_results/isp_resize/detections/gpu_nhcw_b8/detections.jsonl` |
| `73f264788404a66638c01ef1dd48520e` | `07_results/isp_resize/detections/isp_A_nhwc_b8/detections.jsonl` | `07_results/isp_resize/detections/isp_A_filter_lanczos3/detections.jsonl`<br>`07_results/isp_resize/detections/isp_C_nhcw_b8/detections.jsonl` |

`dump_gpu/detections.jsonl` 은 크기는 같지만 박스 순서가 달라 바이트 일치가 아니므로 남겼다.

## 기타 파일 대조표

| md5 | 남긴 파일 | 지운 파일 (내용 동일) |
|---|---|---|
| `ca32bf6060eb9f1aff87827b8a7777b5` | `07_results/dmaheap_unify/verification/asan_p0_heapimport.txt` | `07_results/dmaheap_unify/verification/asan_stdout_p0.log` |
