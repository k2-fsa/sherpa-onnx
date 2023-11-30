include_directories(${CMAKE_SOURCE_DIR})

if(SHERPA_ONNX_ENABLE_PYTHON)
  message(STATUS "PYTHON_EXECUTABLE: ${PYTHON_EXECUTABLE}")
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "import sys; print('.'.join(sys.version.split('.')[:2]))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_VERSION
  )
  message(STATUS "PYTHON_VERSION: ${PYTHON_VERSION}")
endif()

set(sources
  base64-decode.cc
  cat.cc
  circular-buffer.cc
  context-graph.cc
  endpoint.cc
  features.cc
  file-utils.cc
  hypothesis.cc
  offline-ctc-fst-decoder-config.cc
  offline-ctc-fst-decoder.cc
  offline-ctc-greedy-search-decoder.cc
  offline-ctc-model.cc
  offline-lm-config.cc
  offline-lm.cc
  offline-model-config.cc
  offline-nemo-enc-dec-ctc-model-config.cc
  offline-nemo-enc-dec-ctc-model.cc
  offline-paraformer-greedy-search-decoder.cc
  offline-paraformer-model-config.cc
  offline-paraformer-model.cc
  offline-recognizer-impl.cc
  offline-recognizer.cc
  offline-rnn-lm.cc
  offline-stream.cc
  offline-tdnn-ctc-model.cc
  offline-tdnn-model-config.cc
  offline-transducer-greedy-search-decoder.cc
  offline-transducer-model-config.cc
  offline-transducer-model.cc
  offline-transducer-modified-beam-search-decoder.cc
  offline-wenet-ctc-model-config.cc
  offline-wenet-ctc-model.cc
  offline-whisper-greedy-search-decoder.cc
  offline-whisper-model-config.cc
  offline-whisper-model.cc
  offline-zipformer-ctc-model-config.cc
  offline-zipformer-ctc-model.cc
  online-conformer-transducer-model.cc
  online-ctc-greedy-search-decoder.cc
  online-ctc-model.cc
  online-lm-config.cc
  online-lm.cc
  online-lstm-transducer-model.cc
  online-model-config.cc
  online-paraformer-model-config.cc
  online-paraformer-model.cc
  online-recognizer-impl.cc
  online-recognizer.cc
  online-rnn-lm.cc
  online-stream.cc
  online-transducer-decoder.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  online-transducer-modified-beam-search-decoder.cc
  online-wenet-ctc-model-config.cc
  online-wenet-ctc-model.cc
  online-zipformer-transducer-model.cc
  online-zipformer2-transducer-model.cc
  onnx-utils.cc
  packed-sequence.cc
  pad-sequence.cc
  parse-options.cc
  piper-phonemize-lexicon.cc
  provider.cc
  resample.cc
  session.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  slice.cc
  stack.cc
  symbol-table.cc
  text-utils.cc
  transpose.cc
  unbind.cc
  utils.cc
  vad-model-config.cc
  vad-model.cc
  voice-activity-detector.cc
  wave-reader.cc
)

list(APPEND sources
  lexicon.cc
  offline-tts-impl.cc
  offline-tts-model-config.cc
  offline-tts-vits-model-config.cc
  offline-tts-vits-model.cc
  offline-tts.cc
  wave-writer.cc
)

if(SHERPA_ONNX_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()
add_library(sherpa-onnx-core ${sources})
if(APPLE)
  target_compile_options(sherpa-onnx-core PRIVATE
    -Wno-deprecated-declarations
  )
endif()

if(NOT WIN32)
  target_link_libraries(sherpa-onnx-core -pthread)
endif()

if(ANDROID_NDK)
  target_link_libraries(sherpa-onnx-core android log)
endif()

target_link_libraries(sherpa-onnx-core kaldi-native-fbank-core)

target_link_libraries(sherpa-onnx-core kaldi-decoder-core)

if(BUILD_SHARED_LIBS)
  target_link_libraries(sherpa-onnx-core onnxruntime)
else()
  target_link_libraries(sherpa-onnx-core ${onnxruntime_lib_files})
endif()

if(NOT BUILD_SHARED_LIBS AND APPLE)
  target_link_libraries(sherpa-onnx-core "-framework Foundation")
endif()

if(SHERPA_ONNX_ENABLE_GPU)
  target_link_libraries(sherpa-onnx-core
    onnxruntime_providers_cuda
    onnxruntime_providers_shared
  )
endif()

target_link_libraries(sherpa-onnx-core piper_phonemize)

if(SHERPA_ONNX_ENABLE_CHECK)
  target_compile_definitions(sherpa-onnx-core PUBLIC SHERPA_ONNX_ENABLE_CHECK=1)

  if(SHERPA_ONNX_HAVE_EXECINFO_H)
    target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_HAVE_EXECINFO_H=1)
  endif()

  if(SHERPA_ONNX_HAVE_CXXABI_H)
    target_compile_definitions(sherpa-onnx-core PRIVATE SHERPA_ONNX_HAVE_CXXABI_H=1)
  endif()
endif()

if(NOT BUILD_SHARED_LIBS AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-onnx-core -pthread -ldl)
endif()

add_executable(sherpa-onnx sherpa-onnx.cc)
add_executable(sherpa-onnx-offline sherpa-onnx-offline.cc)
add_executable(sherpa-onnx-offline-parallel sherpa-onnx-offline-parallel.cc)
add_executable(sherpa-onnx-offline-tts sherpa-onnx-offline-tts.cc)


target_link_libraries(sherpa-onnx sherpa-onnx-core)
target_link_libraries(sherpa-onnx-offline sherpa-onnx-core)
target_link_libraries(sherpa-onnx-offline-parallel sherpa-onnx-core)
target_link_libraries(sherpa-onnx-offline-tts sherpa-onnx-core)
if(NOT WIN32)
  target_link_libraries(sherpa-onnx "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
  target_link_libraries(sherpa-onnx "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

  target_link_libraries(sherpa-onnx-offline "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
  target_link_libraries(sherpa-onnx-offline "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

  target_link_libraries(sherpa-onnx-offline-parallel "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
  target_link_libraries(sherpa-onnx-offline-parallel "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

  target_link_libraries(sherpa-onnx-offline-tts "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
  target_link_libraries(sherpa-onnx-offline-tts "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

  if(SHERPA_ONNX_ENABLE_PYTHON)
    target_link_libraries(sherpa-onnx "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
    target_link_libraries(sherpa-onnx-offline "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
    target_link_libraries(sherpa-onnx-offline-parallel "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
    target_link_libraries(sherpa-onnx-offline-tts "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
  endif()
endif()

if(SHERPA_ONNX_ENABLE_PYTHON AND WIN32)
  install(TARGETS sherpa-onnx-core DESTINATION ..)
else()
  install(TARGETS sherpa-onnx-core DESTINATION lib)
endif()

if(WIN32 AND BUILD_SHARED_LIBS)
  install(TARGETS sherpa-onnx-core DESTINATION bin)
endif()

install(
  TARGETS
    sherpa-onnx
    sherpa-onnx-offline
    sherpa-onnx-offline-parallel
    sherpa-onnx-offline-tts
  DESTINATION
    bin
)

if(SHERPA_ONNX_HAS_ALSA)
  add_executable(sherpa-onnx-alsa sherpa-onnx-alsa.cc alsa.cc)
  target_link_libraries(sherpa-onnx-alsa sherpa-onnx-core)

  if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
    target_link_libraries(sherpa-onnx-alsa -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(sherpa-onnx-alsa asound)
  endif()
  install(TARGETS sherpa-onnx-alsa DESTINATION bin)
endif()

if(SHERPA_ONNX_ENABLE_PORTAUDIO)
  add_executable(sherpa-onnx-microphone
    sherpa-onnx-microphone.cc
    microphone.cc
  )

  add_executable(sherpa-onnx-microphone-offline
    sherpa-onnx-microphone-offline.cc
    microphone.cc
  )

  add_executable(sherpa-onnx-vad-microphone
    sherpa-onnx-vad-microphone.cc
    microphone.cc
  )

  add_executable(sherpa-onnx-vad-microphone-offline-asr
    sherpa-onnx-vad-microphone-offline-asr.cc
    microphone.cc
  )

  if(BUILD_SHARED_LIBS)
    set(PA_LIB portaudio)
  else()
    set(PA_LIB portaudio_static)
  endif()

  set(exes
    sherpa-onnx-microphone
    sherpa-onnx-microphone-offline
    sherpa-onnx-vad-microphone
    sherpa-onnx-vad-microphone-offline-asr
  )
  foreach(exe IN LISTS exes)
    target_link_libraries(${exe} ${PA_LIB} sherpa-onnx-core)
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")
    endforeach()

    if(SHERPA_ONNX_ENABLE_PYTHON)

      foreach(exe IN LISTS exes)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endforeach()
    endif()
  endif()

  install(
    TARGETS ${exes}
    DESTINATION
      bin
  )
endif()

if(SHERPA_ONNX_ENABLE_WEBSOCKET)
  add_definitions(-DASIO_STANDALONE)
  add_definitions(-D_WEBSOCKETPP_CPP11_STL_)

  add_executable(sherpa-onnx-online-websocket-server
    online-websocket-server-impl.cc
    online-websocket-server.cc
  )
  target_link_libraries(sherpa-onnx-online-websocket-server sherpa-onnx-core)

  add_executable(sherpa-onnx-online-websocket-client
    online-websocket-client.cc
  )
  target_link_libraries(sherpa-onnx-online-websocket-client sherpa-onnx-core)

  if(NOT WIN32)
    target_link_libraries(sherpa-onnx-online-websocket-server -pthread)
    target_compile_options(sherpa-onnx-online-websocket-server PRIVATE -Wno-deprecated-declarations)

    target_link_libraries(sherpa-onnx-online-websocket-client -pthread)
    target_compile_options(sherpa-onnx-online-websocket-client PRIVATE -Wno-deprecated-declarations)
  endif()

  # For offline websocket
  add_executable(sherpa-onnx-offline-websocket-server
    offline-websocket-server-impl.cc
    offline-websocket-server.cc
  )
  target_link_libraries(sherpa-onnx-offline-websocket-server sherpa-onnx-core)

  if(NOT WIN32)
    target_link_libraries(sherpa-onnx-offline-websocket-server -pthread)
    target_compile_options(sherpa-onnx-offline-websocket-server PRIVATE -Wno-deprecated-declarations)
  endif()

  if(NOT WIN32)
    target_link_libraries(sherpa-onnx-online-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-onnx-online-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    target_link_libraries(sherpa-onnx-online-websocket-client "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-onnx-online-websocket-client "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    target_link_libraries(sherpa-onnx-offline-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-onnx-offline-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    if(SHERPA_ONNX_ENABLE_PYTHON)
      target_link_libraries(sherpa-onnx-online-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      target_link_libraries(sherpa-onnx-online-websocket-client "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      target_link_libraries(sherpa-onnx-offline-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
    endif()
  endif()

  install(
    TARGETS
      sherpa-onnx-online-websocket-server
      sherpa-onnx-online-websocket-client
      sherpa-onnx-offline-websocket-server
    DESTINATION
      bin
  )
endif()


if(SHERPA_ONNX_ENABLE_TESTS)
  set(sherpa_onnx_test_srcs
    cat-test.cc
    circular-buffer-test.cc
    context-graph-test.cc
    packed-sequence-test.cc
    pad-sequence-test.cc
    piper-phonemize-test.cc
    slice-test.cc
    stack-test.cc
    transpose-test.cc
    unbind-test.cc
    utfcpp-test.cc
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
