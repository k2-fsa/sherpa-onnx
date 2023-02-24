include_directories(${CMAKE_SOURCE_DIR})
add_library(sherpa-onnx-c-api c-api.cc)
target_link_libraries(sherpa-onnx-c-api sherpa-onnx-core)

install(TARGETS sherpa-onnx-c-api DESTINATION lib)

install(FILES c-api.h
  DESTINATION include/sherpa-onnx/c-api
)

