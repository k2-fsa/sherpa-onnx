include_directories(${CMAKE_SOURCE_DIR})

add_library(sherpa-onnx-core
  cat.cc
  features.cc
  online-lstm-transducer-model.cc
  online-recognizer.cc
  online-stream.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  online-zipformer-transducer-model.cc
  onnx-utils.cc
  symbol-table.cc
  text-utils.cc
  unbind.cc
  wave-reader.cc
)

target_link_libraries(sherpa-onnx-core
  onnxruntime
  kaldi-native-fbank-core
)

add_executable(sherpa-onnx sherpa-onnx.cc)

target_link_libraries(sherpa-onnx sherpa-onnx-core)
if(NOT WIN32)
  target_link_libraries(sherpa-onnx "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
endif()

install(TARGETS sherpa-onnx-core DESTINATION lib)
install(TARGETS sherpa-onnx DESTINATION bin)

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
