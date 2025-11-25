include_directories(${PROJECT_SOURCE_DIR})
add_library(sherpa-onnx-c-api c-api.cc)
target_link_libraries(sherpa-onnx-c-api sherpa-onnx-core)
target_include_directories(sherpa-onnx-c-api PUBLIC ${PROJECT_SOURCE_DIR})

if(BUILD_SHARED_LIBS)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_BUILD_SHARED_LIBS=1)
  target_compile_definitions(sherpa-onnx-c-api PUBLIC SHERPA_ONNX_BUILD_MAIN_LIB=1)
endif()

add_library(sherpa-onnx-cxx-api cxx-api.cc)
target_link_libraries(sherpa-onnx-cxx-api sherpa-onnx-c-api)
target_include_directories(sherpa-onnx-cxx-api PUBLIC ${PROJECT_SOURCE_DIR})

if(UNIX AND NOT APPLE)
  set(EXPORTED_SYMBOLS_FILE_C "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c.lds")
  set(LINK_FLAGS_C "-Wl,--version-script=${EXPORTED_SYMBOLS_FILE_C}")
elseif(APPLE)
  set(EXPORTED_SYMBOLS_FILE_C "${CMAKE_CURRENT_SOURCE_DIR}/sherpa-onnx-symbols-c.exp")
  set(LINK_FLAGS_C "-Wl,-exported_symbols_list,${EXPORTED_SYMBOLS_FILE_C}")
endif()

if(DEFINED LINK_FLAGS_C)
  set_target_properties(sherpa-onnx-c-api PROPERTIES
    LINK_FLAGS "${LINK_FLAGS_C}"
  )
endif()

install(
  TARGETS
    sherpa-onnx-c-api
    sherpa-onnx-cxx-api
  DESTINATION
    lib
)

install(
  FILES
    c-api.h
    cxx-api.h
  DESTINATION
    include/sherpa-onnx/c-api
)
