include_directories(${CMAKE_SOURCE_DIR})

set(sources
  cat.cc
  endpoint.cc
  features.cc
  online-lstm-transducer-model.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  online-zipformer-transducer-model.cc
  onnx-utils.cc
  parse-options.cc
  resample.cc
  symbol-table.cc
  text-utils.cc
  unbind.cc
  wave-reader.cc
)

if(SHERPA_ONNX_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()

add_library(sherpa-onnx-core ${sources})

target_link_libraries(sherpa-onnx-core
  onnxruntime
  kaldi-native-fbank-core
)

if(SHERPA_ONNX_ENABLE_CHECK)
  target_compile_definitions(sherpa-onnx-core PUBLIC SHERPA_ONNX_ENABLE_CHECK=1)

  if(SHERPA_ONNX_HAVE_EXECINFO_H)
    target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_HAVE_EXECINFO_H=1)
  endif()

  if(SHERPA_ONNX_HAVE_CXXABI_H)
    target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_HAVE_CXXABI_H=1)
  endif()
endif()

add_executable(sherpa-onnx sherpa-onnx.cc)

target_link_libraries(sherpa-onnx sherpa-onnx-core)
if(NOT WIN32)
  target_link_libraries(sherpa-onnx "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
endif()

install(TARGETS sherpa-onnx-core DESTINATION lib)
install(TARGETS sherpa-onnx DESTINATION bin)

if(SHERPA_ONNX_HAS_ALSA)
  add_executable(sherpa-onnx-alsa sherpa-onnx-alsa.cc alsa.cc)
  target_link_libraries(sherpa-onnx-alsa PRIVATE sherpa-onnx-core)

  if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
    target_link_libraries(sherpa-onnx-alsa PRIVATE -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(sherpa-onnx-alsa PRIVATE asound)
  endif()
  install(TARGETS sherpa-onnx-alsa DESTINATION bin)
endif()

if(SHERPA_ONNX_ENABLE_PORTAUDIO)
  add_executable(sherpa-onnx-microphone
    sherpa-onnx-microphone.cc
    microphone.cc
  )

  if(BUILD_SHARED_LIBS)
    set(PA_LIB portaudio)
  else()
    set(PA_LIB portaudio_static)
  endif()

  target_link_libraries(sherpa-onnx-microphone PRIVATE ${PA_LIB} sherpa-onnx-core)

  install(TARGETS sherpa-onnx-microphone DESTINATION bin)
endif()


if(SHERPA_ONNX_ENABLE_TESTS)
  set(sherpa_onnx_test_srcs
    cat-test.cc
    unbind-test.cc
  )

  function(sherpa_onnx_add_test source)
    get_filename_component(name ${source} NAME_WE)
    set(target_name ${name})
    add_executable(${target_name} "${source}")

    target_link_libraries(${target_name}
      PRIVATE
        gtest
        gtest_main
        sherpa-onnx-core
    )

    add_test(NAME "${target_name}"
      COMMAND
        $<TARGET_FILE:${target_name}>
    )
  endfunction()

  foreach(source IN LISTS sherpa_onnx_test_srcs)
    sherpa_onnx_add_test(${source})
  endforeach()
endif()
