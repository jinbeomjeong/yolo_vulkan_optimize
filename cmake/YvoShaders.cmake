# 셰이더 컴파일 헬퍼 (glslc: *.comp -> *.spv -> 임베드 헤더)
#   yvo_add_shaders(<target> <shader>...)
#   - glslc --target-env=vulkan1.3 -O -Werror 로 컴파일, -MD 의존성 파일로 #include 변경도 추적
#   - spirv-val --target-env vulkan1.3 로 검증 (실패하면 빌드 실패)
#   - 산출물: ${CMAKE_CURRENT_BINARY_DIR}/<name>.spv, generated/<name>_spv.h
#     (배열 이름: k<CamelName>Spv, 예: sand_letterbox.comp -> kSandLetterboxSpv)
#   - <target>은 INTERFACE 라이브러리. 링크하면 generated/가 include 경로에 추가되고 빌드 순서가 보장된다.
find_program(YVO_GLSLC glslc REQUIRED)
find_program(YVO_SPIRV_VAL spirv-val REQUIRED)
function(yvo_add_shaders target)
  set(gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${gen_dir})
  set(headers "")
  foreach(src IN LISTS ARGN)
    get_filename_component(abs ${src} ABSOLUTE)
    get_filename_component(name ${src} NAME_WE)
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
    set(hdr ${gen_dir}/${name}_spv.h)
    # snake_case -> CamelCase 배열 이름
    string(REPLACE "_" ";" parts ${name})
    set(sym "k")
    foreach(part IN LISTS parts)
      string(SUBSTRING ${part} 0 1 first)
      string(SUBSTRING ${part} 1 -1 rest)
      string(TOUPPER ${first} first)
      string(APPEND sym "${first}${rest}")
    endforeach()
    string(APPEND sym "Spv")
    add_custom_command(
      OUTPUT ${spv}
      COMMAND ${YVO_GLSLC} --target-env=vulkan1.3 -O -Werror -MD -MF ${spv}.d -MT ${spv} -o ${spv} ${abs}
      COMMAND ${YVO_SPIRV_VAL} --target-env vulkan1.3 ${spv}
      DEPENDS ${abs}
      DEPFILE ${spv}.d
      COMMENT "glslc + spirv-val ${src}"
      VERBATIM)
    add_custom_command(
      OUTPUT ${hdr}
      COMMAND ${CMAKE_COMMAND} -DSPV=${spv} -DOUT=${hdr} -DSYMBOL=${sym}
              -P ${PROJECT_SOURCE_DIR}/cmake/yvo_embed_spirv.cmake
      DEPENDS ${spv} ${PROJECT_SOURCE_DIR}/cmake/yvo_embed_spirv.cmake
      COMMENT "embed ${name}.spv -> ${name}_spv.h"
      VERBATIM)
    list(APPEND headers ${hdr})
  endforeach()
  add_custom_target(${target}_gen DEPENDS ${headers})
  add_library(${target} INTERFACE)
  target_include_directories(${target} INTERFACE ${gen_dir})
  add_dependencies(${target} ${target}_gen)
endfunction()
