
add_executable(c-api-alsa c-api-alsa.cc alsa.cc)
target_link_libraries(c-api-alsa sherpa-onnx-c-api cargs)

add_executable(hlg-c-api-alsa hlg-c-api-alsa.cc alsa.cc)
target_link_libraries(hlg-c-api-alsa sherpa-onnx-c-api cargs)

set(exes
  c-api-alsa
  hlg-c-api-alsa
)

foreach(exe IN LISTS exes)
  if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
    target_link_libraries(${exe} -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(${exe} asound)
  endif()
endforeach()
