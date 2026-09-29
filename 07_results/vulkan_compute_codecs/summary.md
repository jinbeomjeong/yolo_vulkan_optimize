# FFmpeg Vulkan compute 코덱 × V3DV (Pi 5) 프로브

프로젝트 본류와 다른 방향의 조사다. "Pi 5에서 영상 디코드를 Vulkan(GPU)으로 할 수 있는가"를 확인한다.
V3DV에는 Vulkan Video 확장이 없다. 그래서 FFmpeg 8.0부터 들어온 **Vulkan compute 코덱**(전용 비디오 엔진 없이 compute 셰이더만 쓰는 코덱)이 유일한 후보다.

## 측정 조건
- 일시: 2026-09-30, Raspberry Pi CM5
- 시스템: 커널 6.18.50+rpt-rpi-2712, 부트로더 2026/09/25(`v3d_freq=1150`)
- GPU: V3D 7.1.10.2, Mesa 26.2.2 V3DV (Vulkan 1.3). 비교 대상은 llvmpipe(같은 Mesa, CPU로 도는 Vulkan)다.
- FFmpeg 8.1.3 릴리스 tarball(sha256·GPG 확인)을 **저장소 밖에서 정적 빌드**했다.
  - configure: `--enable-vulkan --enable-libshaderc --enable-libdrm`
  - 시스템 FFmpeg 7.1.5 rpt2는 건드리지 않았다.
  - 바이너리는 둘이다: 원본(pristine)과, semaphore 조회를 한 줄 고친 연구용 패치(semfix).
- 입력: `04_data/videos/video_h265.mp4`의 앞 300프레임을 CPU로 인코드했다.
  - FFv1 level 3, 24 slices(기본 Golomb, `-coder 1`로 range)
  - ProRes `prores_ks -profile:v 2`, 4:2:2 10-bit
  - DPX 30장(rgb24, gbrp10)
- 반복: 3회, 중앙값(V3D FFv1만 1회·5프레임). 각 실행은 SoC 75 °C 미만에서 시작했다. NPU는 쓰지 않았다.
- 환경은 [`env_bench.txt`](env_bench.txt), 근거는 [`evidence/`](evidence/)에 있다(원문 에러, gdb 백트레이스, SPIR-V capability 대조, framemd5, gpu_stats, 온도 추적).

## 표
| 포맷 | CPU fps / CPU ms·f | V3D (semfix) | llvmpipe fps / CPU ms·f | llvmpipe md5 = CPU |
|---|---|---|---|---|
| FFv1 Golomb 디코드 | 82.6 / 42.3 | 셰이더 컴파일러 abort (`Unknown NIR ALU inst: 64 … u2u64`) | 크래시 (JIT SIGSEGV) | – |
| FFv1 range 디코드 | 57.9 / 62.8 | 실행되지만 **출력 틀림** (0/10, 24개 슬라이스 모두 overread), GPU 약 20 s/f | 0.49 / 7833 | 30/30 동일 |
| ProRes 422 10-bit 디코드 | 148.5 / 23.3 | VLD 파이프라인 생성이 11분 넘게 끝나지 않음 | 12.3 / 215 | ±1 LSB (PSNR ≥ 76.6 dB) |
| DPX rgb24 (copy) 디코드 | 235 / 7.9 | 47.5 fps지만 R 채널만 기록 (G = B = 0) | 24.5 / 133 | 30/30 동일 |
| DPX gbrp10 (unpack) 디코드 | 99.5 / 16.1 | 41.9 fps지만 출력 틀림 | 42.2 / 66 | 30/30 동일 |
| `ffv1_vulkan` 인코드 | – | 비트스트림 손상 | 무손실 (10/10) | – |
| `prores_ks_vulkan` 인코드 | – | 셰이더 컴파일러 abort (`i2i64`) | 손상 (PSNR 11.7 dB) | – |

**원본 8.1.3과 시스템 7.1.5는 V3D에서 Vulkan 프레임을 하나도 할당하지 못한다(hwupload 포함).** 모두 `Failed to create semaphore: VK_ERROR_INVALID_EXTERNAL_HANDLE`로 실패한다.

## 결론
- **V3DV는 FFmpeg Vulkan compute 코덱을 실행할 수 없다.** 대상은 FFv1·ProRes·ProRes RAW·DPX와 인코더까지 전부다. 막히는 층은 셋이다.
  1. **FFmpeg 버그**(시스템 7.1.5 포함)
     - semaphore를 외부로 내보낼 수 있는지 binary semaphore 기준으로 조회해 놓고, 실제로는 timeline semaphore를 내보내기 가능하게 만든다. 이는 `VUID-VkExportSemaphoreCreateInfo-handleTypes-01124` 위반이다.
     - V3D는 binary만 내보낼 수 있다(C 프로브로 확인).
     - 한 줄 패치로 우회된다. upstream에 보고할 만한 결함이다.
  2. **드라이버 기능 부족**(본질적 원인)
     - 코덱 셰이더가 선언하는 기능 세 가지를 V3D는 모두 지원하지 않는다: `Int64`(`shaderInt64=false`), `StorageImageWriteWithoutFormat`(false), `RuntimeDescriptorArray`와 non-uniform indexing(descriptor indexing false).
     - 증상은 컴파일러 abort, 컴파일 행, 그리고 **조용한 오동작**이다. Int64를 쓰지 않는 DPX copy 셰이더도 포맷 없는 store 때문에 R 채널만 쓴다.
  3. **FFmpeg가 이 기능들을 확인하지 않고 파이프라인을 만든다.** 이는 `VUID-VkShaderModuleCreateInfo-pCode-08740` 위반이다.
- **셰이더 로직은 대체로 맞다.** llvmpipe에서 FFv1 range·DPX는 바이트 동일이고, ProRes는 ±1 LSB(float IDCT)다. 따라서 V3D 실패의 책임은 드라이버 기능 부족에 있다.
  - 예외: FFv1 Golomb 디코드와 `prores_ks_vulkan` 인코드는 llvmpipe에서도 깨진다. FFmpeg 쪽인지 llvmpipe 쪽인지는 가르지 못했다.
- **돌더라도 성능 이점은 없다.** llvmpipe는 CPU 디코더보다 12~170배 느리다. V3D DPX도 CPU의 1/5 수준이고, FFv1은 GPU 시간만 프레임당 약 20초였다.
- **Pi 5의 GPU 기반 영상 디코드에 대한 의미**
  - V3DV에는 Vulkan Video(HEVC `-hwaccel vulkan`)도 없고, compute 코덱의 전제 기능도 없다. FFmpeg의 Vulkan 디코드 경로는 전부 막혀 있다.
  - 성립하는 구조는 지금 프로젝트의 구조 하나다: HEVC는 전용 HW 디코더(DRM_PRIME/V4L2)가 맡고, V3D는 자체 int32 셰이더로 전·후처리를 맡는다.
  - 자체 GPU 셰이더를 만들 때는 64비트 정수와 포맷 없는 image store를 피해야 한다.

## 주의·남은 일
- Vulkan 경로의 FPS에는 장치 초기화와 셰이더 컴파일(JIT) 시간이 들어 있다. 30프레임 측정에서는 이 시작 비용의 비중이 크다.
- llvmpipe FFv1 2·3회차는 SoC 소프트 온도 제한(`0xe0008`, ARM 1.5~2.4 GHz) 중에 쟀다([`evidence/thermal_trace.txt`](evidence/thermal_trace.txt)). 결론(초당 1프레임 미만)은 바뀌지 않는다.
- V3D 행은 연구용 패치 바이너리(semfix) 결과다. 원본으로는 셰이더 단계에 도달하지 못한다. 패치 diff는 `evidence/blocker1_timeline_semaphore_export.txt`에 있고, 빌드는 저장소 밖 `/home/pi/build-ffmpeg-upstream/`(약 2.1 GB)에 있다.
- ProRes RAW는 입력 샘플이 없어 실행하지 않았다. 셰이더가 Int64와 포맷 없는 store를 선언하므로 같은 결과를 예상한다.
- Mesa 디버그 심볼이 없어 ProRes 컴파일 행이 Mesa 내부 어디서 멈추는지는 특정하지 못했다.
- 후속 후보: FFmpeg semaphore 버그 upstream 보고, V3DV의 `shaderInt64`와 포맷 없는 store 지원 추적(Mesa).
