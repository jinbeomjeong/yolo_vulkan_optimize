# Shader compile helper (glslc: *.comp -> *.spv -> embedded header)
#   yvo_add_shaders(<target> <shader>...)
#   - compiled with glslc --target-env=vulkan1.3 -O -Werror; a -MD dependency file also tracks #include changes
#   - validated with spirv-val --target-env vulkan1.3 (the build fails on failure)
#   - outputs: ${CMAKE_CURRENT_BINARY_DIR}/<name>.spv, generated/<name>_spv.h
#     (array name: k<CamelName>Spv, e.g. sand_letterbox.comp -> kSandLetterboxSpv)
#   - <target> is an INTERFACE library. Linking it adds generated/ to the include path and guarantees build order.
#   - variants: writing "<src>.comp@<out_name>@<DEF1>[,<DEF2>...]" compiles the same source once more with -D<DEF>
#     and produces <out_name>.spv / <out_name>_spv.h (e.g. overlay_boxes.comp@overlay_boxes_buf@YVO_OVERLAY_BUFFER).
#     Definitions whose value contains a comma (FOO=a,b) cannot be expressed.
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
        message(FATAL_ERROR "yvo_add_shaders: a variant entry must have the form <src>@<out_name>@<DEFS>: ${entry}")
      endif()
      list(GET parts 0 src)
      list(GET parts 1 name)
      list(GET parts 2 def_csv)
      if(NOT name MATCHES "^[A-Za-z][A-Za-z0-9_]*$")
        message(FATAL_ERROR "yvo_add_shaders: out_name must match [A-Za-z][A-Za-z0-9_]*: ${entry}")
      endif()
      if(def_csv STREQUAL "")
        message(FATAL_ERROR "yvo_add_shaders: DEFS of the variant is empty: ${entry}")
      endif()
      string(REPLACE "," ";" def_list "${def_csv}")
      foreach(d IN LISTS def_list)
        if(d STREQUAL "")
          # glslc prints an error for a -D without an argument yet returns exit code 0 → block it here
          message(FATAL_ERROR "yvo_add_shaders: DEFS contains an empty item (check the commas): ${entry}")
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
      message(FATAL_ERROR "yvo_add_shaders(${target}): output name '${name}' is duplicated: ${entry}")
    endif()
    list(APPEND seen_names ${name})
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/${name}.spv)
    set(hdr ${gen_dir}/${name}_spv.h)
    # snake_case -> CamelCase array name
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
