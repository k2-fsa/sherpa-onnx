include_directories(${CMAKE_SOURCE_DIR})

if(NOT DEFINED ANDROID_ABI)
  if(NOT DEFINED ENV{JAVA_HOME})
    message(FATAL_ERROR "Please set the environment variable JAVA_HOME")
  endif()
  include_directories($ENV{JAVA_HOME}/include)
  include_directories($ENV{JAVA_HOME}/include/linux)
  include_directories($ENV{JAVA_HOME}/include/darwin)
endif()

add_library(sherpa-onnx-jni
  audio-tagging.cc
  jni.cc
  offline-stream.cc
)
target_link_libraries(sherpa-onnx-jni sherpa-onnx-core)
install(TARGETS sherpa-onnx-jni DESTINATION lib)
