cmake_minimum_required(VERSION 3.15)
project(kws_sdk LANGUAGES CXX)
# 对外只暴露 kws_api.h，实现依赖 sherpa-onnx CXX API
add_library(kws_sdk STATIC src/kws_api.cc)
target_include_directories(kws_sdk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(kws_sdk PUBLIC sherpa-onnx-cxx-api)

# 编译本 target 时能找到 kws_api.h 和 sherpa-onnx 头文件
target_include_directories(kws_sdk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS kws_sdk
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)
install(DIRECTORY include/
  DESTINATION include
  FILES_MATCHING PATTERN "*.h")
