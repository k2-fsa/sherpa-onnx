include_directories(${CMAKE_SOURCE_DIR})

pybind11_add_module(_sherpa_onnx
  display.cc
  endpoint.cc
  features.cc
  offline-model-config.cc
  offline-paraformer-model-config.cc
  offline-recognizer.cc
  offline-stream.cc
  offline-transducer-model-config.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-model-config.cc
  sherpa-onnx.cc
)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_sherpa_onnx PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

if(NOT WIN32)
  target_link_libraries(_sherpa_onnx PRIVATE "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/sherpa_onnx/lib")
endif()

target_link_libraries(_sherpa_onnx PRIVATE sherpa-onnx-core)

install(TARGETS _sherpa_onnx
  DESTINATION ../
)
