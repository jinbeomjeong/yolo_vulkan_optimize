# SPIR-V 바이너리 → C++ 헤더(uint32_t 배열) 변환. cmake -P 로 실행한다.
#   -DSPV=<입력 .spv> -DOUT=<출력 .h> -DSYMBOL=<배열 이름>
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
# 바이트 4개(리틀 엔디언) → 32bit 워드
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
"// 자동 생성 파일: ${spv_name} (yvo_embed_spirv.cmake). 수정하지 말 것.\n"
"#pragma once\n#include <cstdint>\n\n"
"inline constexpr std::uint32_t ${SYMBOL}[] = {\n    ${body}\n};\n")
# 내용이 같으면 타임스탬프를 바꾸지 않아 불필요한 재컴파일을 막는다
file(COPY_FILE "${OUT}.tmp" "${OUT}" ONLY_IF_DIFFERENT)
file(REMOVE "${OUT}.tmp")
