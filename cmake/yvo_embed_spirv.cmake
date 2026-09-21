# SPIR-V binary → C++ header (uint32_t array) conversion. Run with cmake -P.
#   -DSPV=<input .spv> -DOUT=<output .h> -DSYMBOL=<array name>
if(NOT SPV OR NOT OUT OR NOT SYMBOL)
  message(FATAL_ERROR "yvo_embed_spirv: SPV, OUT, SYMBOL are required")
endif()
file(READ "${SPV}" hex HEX)
string(LENGTH "${hex}" hex_len)
math(EXPR nbytes "${hex_len} / 2")
math(EXPR rem "${nbytes} % 4")
if(nbytes EQUAL 0 OR NOT rem EQUAL 0)
  message(FATAL_ERROR "yvo_embed_spirv: ${SPV} size ${nbytes} is not a positive multiple of 4")
endif()
# 4 bytes (little endian) → 32-bit word
string(REGEX MATCHALL "([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])([0-9a-f][0-9a-f])" words "${hex}")
set(body "")
set(i 0)
foreach(w IN LISTS words)
  string(SUBSTRING "${w}" 0 2 b0)
  string(SUBSTRING "${w}" 2 2 b1)
  string(SUBSTRING "${w}" 4 2 b2)
  string(SUBSTRING "${w}" 6 2 b3)
  string(APPEND body "0x${b3}${b2}${b1}${b0}u,")
  math(EXPR i "${i} + 1")
  math(EXPR col "${i} % 8")
  if(col EQUAL 0)
    string(APPEND body "\n    ")
  else()
    string(APPEND body " ")
  endif()
endforeach()
get_filename_component(spv_name "${SPV}" NAME)
file(WRITE "${OUT}.tmp"
"// Auto-generated file: ${spv_name} (yvo_embed_spirv.cmake). Do not edit.\n"
"#pragma once\n#include <cstdint>\n\n"
"inline constexpr std::uint32_t ${SYMBOL}[] = {\n    ${body}\n};\n")
# If the content is unchanged, keep the timestamp to avoid needless recompilation
file(COPY_FILE "${OUT}.tmp" "${OUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUT}.tmp")
