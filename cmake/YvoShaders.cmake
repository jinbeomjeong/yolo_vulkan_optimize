# 셰이더 컴파일 헬퍼 (glslc: *.comp -> *.spv -> 임베드 헤더)
#   yvo_add_shaders(<target> <shader>...)
#   - glslc --target-env=vulkan1.3 -O -Werror 로 컴파일, -MD 의존성 파일로 #include 변경도 추적
#   - spirv-val --target-env vulkan1.3 로 검증 (실패하면 빌드 실패)
#   - 산출물: ${CMAKE_CURRENT_BINARY_DIR}/<name>.spv, generated/<name>_spv.h
#     (배열 이름: k<CamelName>Spv, 예: sand_letterbox.comp -> kSandLetterboxSpv)
#   - <target>은 INTERFACE 라이브러리. 링크하면 generated/가 include 경로에 추가되고 빌드 순서가 보장된다.
#   - 변형: "<src>.comp@<out_name>@<DEF1>[,<DEF2>...]" 로 쓰면 같은 소스를 -D<DEF> 로 한 번 더 컴파일해
#     <out_name>.spv / <out_name>_spv.h 를 만든다 (예: overlay_boxes.comp@overlay_boxes_buf@YVO_OVERLAY_BUFFER).
#     값에 쉼표가 든 정의(FOO=a,b)는 표현할 수 없다.
find_program(YVO_GLSLC glslc REQUIRED)
find_program(YVO_SPIRV_VAL spirv-val REQUIRED)
function(yvo_add_shaders target)
  set(gen_dir ${CMAKE_CURRENT_BINARY_DIR}/generated)
  file(MAKE_DIRECTORY ${gen_dir})
  set(headers "")
  set(seen_names "")
  foreach(entry IN LISTS ARGN)
    set(defs "")
    if(entry MATCHES "@")
      string(REPLACE "@" ";" parts "${entry}")
      list(LENGTH parts nparts)
      if(NOT nparts EQUAL 3)
        message(FATAL_ERROR "yvo_add_shaders: 변형 항목은 <src>@<out_name>@<DEFS> 형식이어야 한다: ${entry}")
      endif()
      list(GET parts 0 src)
      list(GET parts 1 name)
      list(GET parts 2 def_csv)
      if(NOT name MATCHES "^[A-Za-z][A-Za-z0-9_]*$")
        message(FATAL_ERROR "yvo_add_shaders: out_name 은 [A-Za-z][A-Za-z0-9_]* 이어야 한다: ${entry}")
      endif()
      if(def_csv STREQUAL "")
        message(FATAL_ERROR "yvo_add_shaders: 변형에 DEFS 가 비어 있다: ${entry}")
      endif()
      string(REPLACE "," ";" def_list "${def_csv}")
      foreach(d IN LISTS def_list)
        if(d STREQUAL "")
          # glslc 는 인자 없는 -D 에 에러를 찍고도 종료 코드 0 을 돌려준다 → 여기서 막는다
          message(FATAL_ERROR "yvo_add_shaders: DEFS 에 빈 항목이 있다(쉼표 확인): ${entry}")
        endif()
        list(APPEND defs "-D${d}")
      endforeach()
      get_filename_component(abs ${src} ABSOLUTE)
    else()
      set(src ${entry})
      get_filename_component(abs ${src} ABSOLUTE)
      get_filename_component(name ${src} NAME_WE)
    endif()
    if(name IN_LIST seen_names)
      message(FATAL_ERROR "yvo_add_shaders(${target}): 출력 이름 '${name}' 이 중복된다: ${entry}")
    endif()
    list(APPEND seen_names ${name})
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
      COMMAND ${YVO_GLSLC} --target-env=vulkan1.3 -O -Werror ${defs} -MD -MF ${spv}.d -MT ${spv} -o ${spv} ${abs}
      COMMAND ${YVO_SPIRV_VAL} --target-env vulkan1.3 ${spv}
      DEPENDS ${abs}
      DEPFILE ${spv}.d
      COMMENT "glslc + spirv-val ${src} -> ${name}.spv"
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
