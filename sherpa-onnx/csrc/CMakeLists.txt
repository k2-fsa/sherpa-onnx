include_directories(${PROJECT_SOURCE_DIR})

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
  bbpe.cc
  cat.cc
  circular-buffer.cc
  context-graph.cc
  endpoint.cc
  features.cc
  file-utils.cc
  fst-utils.cc
  homophone-replacer.cc
  hypothesis.cc
  jieba.cc
  keyword-spotter-impl.cc
  keyword-spotter.cc
  lodr-fst.cc
  offline-canary-model-config.cc
  offline-canary-model.cc
  offline-ctc-fst-decoder-config.cc
  offline-ctc-fst-decoder.cc
  offline-ctc-greedy-search-decoder.cc
  offline-ctc-model.cc
  offline-dolphin-model-config.cc
  offline-dolphin-model.cc
  offline-fire-red-asr-greedy-search-decoder.cc
  offline-fire-red-asr-model-config.cc
  offline-fire-red-asr-model.cc
  offline-lm-config.cc
  offline-lm.cc
  offline-model-config.cc
  offline-moonshine-greedy-search-decoder.cc
  offline-moonshine-model-config.cc
  offline-moonshine-model.cc
  offline-nemo-enc-dec-ctc-model-config.cc
  offline-nemo-enc-dec-ctc-model.cc
  offline-paraformer-greedy-search-decoder.cc
  offline-paraformer-model-config.cc
  offline-paraformer-model.cc
  offline-recognizer-impl.cc
  offline-recognizer.cc
  offline-rnn-lm.cc
  offline-sense-voice-model-config.cc
  offline-sense-voice-model.cc
  offline-source-separation-impl.cc
  offline-source-separation-model-config.cc
  offline-source-separation-spleeter-model-config.cc
  offline-source-separation-spleeter-model.cc
  offline-source-separation-uvr-model-config.cc
  offline-source-separation-uvr-model.cc
  offline-source-separation.cc
  offline-stream.cc
  offline-tdnn-ctc-model.cc
  offline-tdnn-model-config.cc
  offline-telespeech-ctc-model.cc
  offline-transducer-greedy-search-decoder.cc
  offline-transducer-greedy-search-nemo-decoder.cc
  offline-transducer-model-config.cc
  offline-transducer-model.cc
  offline-transducer-modified-beam-search-decoder.cc
  offline-transducer-nemo-model.cc
  offline-wenet-ctc-model-config.cc
  offline-wenet-ctc-model.cc
  offline-whisper-greedy-search-decoder.cc
  offline-whisper-model-config.cc
  offline-whisper-model.cc
  offline-zipformer-ctc-model-config.cc
  offline-zipformer-ctc-model.cc
  online-conformer-transducer-model.cc
  online-ctc-fst-decoder-config.cc
  online-ctc-fst-decoder.cc
  online-ctc-greedy-search-decoder.cc
  online-ctc-model.cc
  online-ebranchformer-transducer-model.cc
  online-lm-config.cc
  online-lm.cc
  online-lstm-transducer-model.cc
  online-model-config.cc
  online-nemo-ctc-model-config.cc
  online-nemo-ctc-model.cc
  online-paraformer-model-config.cc
  online-paraformer-model.cc
  online-recognizer-impl.cc
  online-recognizer.cc
  online-rnn-lm.cc
  online-stream.cc
  online-transducer-decoder.cc
  online-transducer-greedy-search-decoder.cc
  online-transducer-greedy-search-nemo-decoder.cc
  online-transducer-model-config.cc
  online-transducer-model.cc
  online-transducer-modified-beam-search-decoder.cc
  online-transducer-nemo-model.cc
  online-wenet-ctc-model-config.cc
  online-wenet-ctc-model.cc
  online-zipformer-transducer-model.cc
  online-zipformer2-ctc-model-config.cc
  online-zipformer2-ctc-model.cc
  online-zipformer2-transducer-model.cc
  onnx-utils.cc
  packed-sequence.cc
  pad-sequence.cc
  parse-options.cc
  provider-config.cc
  provider.cc
  resample.cc
  session.cc
  silero-vad-model-config.cc
  silero-vad-model.cc
  slice.cc
  spoken-language-identification-impl.cc
  spoken-language-identification.cc
  stack.cc
  symbol-table.cc
  ten-vad-model-config.cc
  ten-vad-model.cc
  text-utils.cc
  transducer-keyword-decoder.cc
  transpose.cc
  unbind.cc
  utils.cc
  vad-model-config.cc
  vad-model.cc
  version.cc
  voice-activity-detector.cc
  wave-reader.cc
  wave-writer.cc
)

# speaker embedding extractor
list(APPEND sources
  speaker-embedding-extractor-impl.cc
  speaker-embedding-extractor-model.cc
  speaker-embedding-extractor-nemo-model.cc
  speaker-embedding-extractor.cc
  speaker-embedding-manager.cc
)

# audio tagging
list(APPEND sources
  audio-tagging-impl.cc
  audio-tagging-label-file.cc
  audio-tagging-model-config.cc
  audio-tagging.cc
  offline-ced-model.cc
  offline-zipformer-audio-tagging-model-config.cc
  offline-zipformer-audio-tagging-model.cc
)

# punctuation
list(APPEND sources
  offline-ct-transformer-model.cc
  offline-punctuation-impl.cc
  offline-punctuation-model-config.cc
  offline-punctuation.cc
  online-cnn-bilstm-model.cc
  online-punctuation-impl.cc
  online-punctuation-model-config.cc
  online-punctuation.cc
)
if(SHERPA_ONNX_ENABLE_RKNN)
  list(APPEND sources
    ./rknn/online-stream-rknn.cc
    ./rknn/online-transducer-greedy-search-decoder-rknn.cc
    ./rknn/online-transducer-modified-beam-search-decoder-rknn.cc
    ./rknn/online-zipformer-ctc-model-rknn.cc
    ./rknn/online-zipformer-transducer-model-rknn.cc
    ./rknn/silero-vad-model-rknn.cc
    ./rknn/utils.cc
  )

endif()

if(SHERPA_ONNX_ENABLE_TTS)
  list(APPEND sources
    hifigan-vocoder.cc
    jieba-lexicon.cc
    kokoro-multi-lang-lexicon.cc
    lexicon.cc
    melo-tts-lexicon.cc
    offline-tts-character-frontend.cc
    offline-tts-frontend.cc
    offline-tts-impl.cc
    offline-tts-kokoro-model-config.cc
    offline-tts-kokoro-model.cc
    offline-tts-matcha-model-config.cc
    offline-tts-matcha-model.cc
    offline-tts-model-config.cc
    offline-tts-vits-model-config.cc
    offline-tts-vits-model.cc
    offline-tts.cc
    piper-phonemize-lexicon.cc
    vocoder.cc
    vocos-vocoder.cc
  )
endif()

list(APPEND sources
  offline-speech-denoiser-gtcrn-model-config.cc
  offline-speech-denoiser-gtcrn-model.cc
  offline-speech-denoiser-impl.cc
  offline-speech-denoiser-model-config.cc
  offline-speech-denoiser.cc
)

if(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION)
  list(APPEND sources
    fast-clustering-config.cc
    fast-clustering.cc
    offline-speaker-diarization-impl.cc
    offline-speaker-diarization-result.cc
    offline-speaker-diarization.cc
    offline-speaker-segmentation-model-config.cc
    offline-speaker-segmentation-pyannote-model-config.cc
    offline-speaker-segmentation-pyannote-model.cc
  )
endif()

if(SHERPA_ONNX_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()

# Always static build
add_library(sherpa-onnx-core STATIC ${sources})


if(WIN32 AND SHERPA_ONNX_LINK_D3D)
    target_link_libraries(sherpa-onnx-core dxguid.lib d3d12.lib dxgi.lib dxcore.lib)
endif()


if(TARGET directml)
    target_link_libraries(sherpa-onnx-core directml)
endif()

set_target_properties(
    sherpa-onnx-core
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    C_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
)

if(APPLE)
  target_compile_options(sherpa-onnx-core PRIVATE
    -Wno-deprecated-declarations
  )
endif()

if(ANDROID_NDK)
  target_link_libraries(sherpa-onnx-core android log)
endif()

target_link_libraries(sherpa-onnx-core
  kaldi-native-fbank-core
  kaldi-decoder-core
  ssentencepiece_core
)
if(DEFINED OHOS AND x${OHOS} STREQUAL xOHOS)
  target_link_libraries(sherpa-onnx-core
    hilog_ndk.z
    rawfile.z
  )
endif()

if(SHERPA_ONNX_ENABLE_GPU)
  target_link_libraries(sherpa-onnx-core
    onnxruntime_providers_shared
  )
endif()

if(SHERPA_ONNX_ENABLE_RKNN)
  if(DEFINED ENV{SHERPA_ONNX_RKNN_TOOLKIT2_LIB_DIR})
    target_link_libraries(sherpa-onnx-core -L$ENV{SHERPA_ONNX_RKNN_TOOLKIT2_LIB_DIR} -lrknnrt)
  else()
    target_link_libraries(sherpa-onnx-core rknnrt)
  endif()
endif()

if((BUILD_SHARED_LIBS OR SHERPA_ONNX_USE_PRE_INSTALLED_ONNXRUNTIME_IF_AVAILABLE) AND NOT DEFINED onnxruntime_lib_files)
  target_link_libraries(sherpa-onnx-core onnxruntime)
else()
  target_link_libraries(sherpa-onnx-core ${onnxruntime_lib_files})
endif()

if(NOT WIN32)
  target_link_libraries(sherpa-onnx-core -lm)
endif()

if(NOT BUILD_SHARED_LIBS AND APPLE)
  target_link_libraries(sherpa-onnx-core "-framework Foundation")
endif()

target_link_libraries(sherpa-onnx-core fstfar fst)

if(SHERPA_ONNX_ENABLE_TTS)
  target_link_libraries(sherpa-onnx-core piper_phonemize)
endif()
target_link_libraries(sherpa-onnx-core cppjieba)

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
  # This is for linux arm32 and arm64
  target_link_libraries(sherpa-onnx-core -ldl)
endif()

if(NOT WIN32 AND NOT SHERPA_ONNX_ENABLE_WASM AND CMAKE_SYSTEM_NAME STREQUAL Linux)
  target_link_libraries(sherpa-onnx-core -pthread)
endif()

if(SHERPA_ONNX_ENABLE_BINARY)
  add_executable(sherpa-onnx sherpa-onnx.cc)
  add_executable(sherpa-onnx-keyword-spotter sherpa-onnx-keyword-spotter.cc)
  add_executable(sherpa-onnx-offline sherpa-onnx-offline.cc)
  add_executable(sherpa-onnx-offline-audio-tagging sherpa-onnx-offline-audio-tagging.cc)
  add_executable(sherpa-onnx-offline-denoiser sherpa-onnx-offline-denoiser.cc)
  add_executable(sherpa-onnx-offline-language-identification sherpa-onnx-offline-language-identification.cc)
  add_executable(sherpa-onnx-offline-parallel sherpa-onnx-offline-parallel.cc)
  add_executable(sherpa-onnx-offline-punctuation sherpa-onnx-offline-punctuation.cc)
  add_executable(sherpa-onnx-offline-source-separation sherpa-onnx-offline-source-separation.cc)
  add_executable(sherpa-onnx-online-punctuation sherpa-onnx-online-punctuation.cc)
  add_executable(sherpa-onnx-version sherpa-onnx-version.cc version.cc)
  add_executable(sherpa-onnx-vad sherpa-onnx-vad.cc)

  if(SHERPA_ONNX_ENABLE_TTS)
    add_executable(sherpa-onnx-offline-tts sherpa-onnx-offline-tts.cc)
  endif()

  if(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION)
    add_executable(sherpa-onnx-offline-speaker-diarization sherpa-onnx-offline-speaker-diarization.cc)
  endif()

  set(main_exes
    sherpa-onnx
    sherpa-onnx-keyword-spotter
    sherpa-onnx-offline
    sherpa-onnx-offline-audio-tagging
    sherpa-onnx-offline-denoiser
    sherpa-onnx-offline-language-identification
    sherpa-onnx-offline-parallel
    sherpa-onnx-offline-punctuation
    sherpa-onnx-offline-source-separation
    sherpa-onnx-online-punctuation
    sherpa-onnx-vad
  )
  if(SHERPA_ONNX_ENABLE_TTS)
    list(APPEND main_exes
      sherpa-onnx-offline-tts
    )
  endif()

  if(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION)
    list(APPEND main_exes
      sherpa-onnx-offline-speaker-diarization
    )
  endif()

  foreach(exe IN LISTS main_exes)
    target_link_libraries(${exe} sherpa-onnx-core)
  endforeach()

  if(NOT WIN32)
    foreach(exe IN LISTS main_exes)
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
      target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

      if(SHERPA_ONNX_ENABLE_PYTHON)
        target_link_libraries(${exe} "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib/python${PYTHON_VERSION}/site-packages/sherpa_onnx/lib")
      endif()
    endforeach()
  endif()
endif()

if(NOT BUILD_SHARED_LIBS)
  install(TARGETS sherpa-onnx-core DESTINATION lib)
endif()

if(SHERPA_ONNX_ENABLE_BINARY)
  install(
    TARGETS
      ${main_exes}
      sherpa-onnx-version
    DESTINATION
      bin
  )
endif()

if(SHERPA_ONNX_HAS_ALSA AND SHERPA_ONNX_ENABLE_BINARY)
  add_executable(sherpa-onnx-alsa sherpa-onnx-alsa.cc alsa.cc)
  add_executable(sherpa-onnx-alsa-offline sherpa-onnx-alsa-offline.cc alsa.cc)
  add_executable(sherpa-onnx-alsa-offline-audio-tagging sherpa-onnx-alsa-offline-audio-tagging.cc alsa.cc)
  add_executable(sherpa-onnx-alsa-offline-speaker-identification sherpa-onnx-alsa-offline-speaker-identification.cc alsa.cc)
  add_executable(sherpa-onnx-keyword-spotter-alsa sherpa-onnx-keyword-spotter-alsa.cc alsa.cc)
  add_executable(sherpa-onnx-vad-alsa sherpa-onnx-vad-alsa.cc alsa.cc)
  add_executable(sherpa-onnx-vad-alsa-offline-asr sherpa-onnx-vad-alsa-offline-asr.cc alsa.cc)


  if(SHERPA_ONNX_ENABLE_TTS)
    add_executable(sherpa-onnx-offline-tts-play-alsa sherpa-onnx-offline-tts-play-alsa.cc alsa-play.cc)
  endif()

  set(exes
    sherpa-onnx-alsa
    sherpa-onnx-alsa-offline
    sherpa-onnx-alsa-offline-speaker-identification
    sherpa-onnx-keyword-spotter-alsa
    sherpa-onnx-vad-alsa
    sherpa-onnx-vad-alsa-offline-asr
    sherpa-onnx-alsa-offline-audio-tagging
  )

  if(SHERPA_ONNX_ENABLE_TTS)
    list(APPEND exes
      sherpa-onnx-offline-tts-play-alsa
    )
  endif()

  #   # To fix the following error for Windows when building exe
  #   #  mismatch detected for 'RuntimeLibrary': value 'MT_StaticRelease' doesn't match value 'MD_Dynamic Release'

  foreach(exe IN LISTS exes)
    target_link_libraries(${exe} sherpa-onnx-core)
  endforeach()

  foreach(exe IN LISTS exes)
    if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
      target_link_libraries(${exe} -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
    else()
      target_link_libraries(${exe} asound)
    endif()
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

if(SHERPA_ONNX_ENABLE_PORTAUDIO AND SHERPA_ONNX_ENABLE_BINARY)
  if(SHERPA_ONNX_ENABLE_TTS)
    add_executable(sherpa-onnx-offline-tts-play
      sherpa-onnx-offline-tts-play.cc
      microphone.cc
    )
  endif()

  add_executable(sherpa-onnx-keyword-spotter-microphone
    sherpa-onnx-keyword-spotter-microphone.cc
    microphone.cc
  )

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

  add_executable(sherpa-onnx-vad-with-offline-asr
    sherpa-onnx-vad-with-offline-asr.cc
  )

  add_executable(sherpa-onnx-vad-microphone-offline-asr
    sherpa-onnx-vad-microphone-offline-asr.cc
    microphone.cc
  )

  add_executable(sherpa-onnx-microphone-offline-speaker-identification
    sherpa-onnx-microphone-offline-speaker-identification.cc
    microphone.cc
  )

  add_executable(sherpa-onnx-microphone-offline-audio-tagging
    sherpa-onnx-microphone-offline-audio-tagging.cc
    microphone.cc
  )

  set(exes
    sherpa-onnx-keyword-spotter-microphone
    sherpa-onnx-microphone
    sherpa-onnx-microphone-offline
    sherpa-onnx-microphone-offline-audio-tagging
    sherpa-onnx-microphone-offline-speaker-identification
    sherpa-onnx-vad-microphone
    sherpa-onnx-vad-microphone-offline-asr
    sherpa-onnx-vad-with-offline-asr
  )
  if(SHERPA_ONNX_ENABLE_TTS)
    list(APPEND exes
      sherpa-onnx-offline-tts-play
    )
  endif()

  foreach(exe IN LISTS exes)
    target_link_libraries(${exe} portaudio_static sherpa-onnx-core)
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

if(SHERPA_ONNX_ENABLE_WEBSOCKET AND SHERPA_ONNX_ENABLE_BINARY)
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
    target_compile_options(sherpa-onnx-online-websocket-server PRIVATE -Wno-deprecated-declarations)

    target_compile_options(sherpa-onnx-online-websocket-client PRIVATE -Wno-deprecated-declarations)
  endif()

  # For offline websocket
  add_executable(sherpa-onnx-offline-websocket-server
    offline-websocket-server-impl.cc
    offline-websocket-server.cc
  )
  target_link_libraries(sherpa-onnx-offline-websocket-server sherpa-onnx-core)

  if(NOT WIN32)
    target_compile_options(sherpa-onnx-offline-websocket-server PRIVATE -Wno-deprecated-declarations)
  endif()

  if(NOT WIN32)
    target_link_libraries(sherpa-onnx-online-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-onnx-online-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    target_link_libraries(sherpa-onnx-online-websocket-client "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-onnx-online-websocket-client "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    target_link_libraries(sherpa-onnx-offline-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../lib")
    target_link_libraries(sherpa-onnx-offline-websocket-server "-Wl,-rpath,${SHERPA_ONNX_RPATH_ORIGIN}/../../../sherpa_onnx/lib")

    if(SHERPA_ONNX_ENABLE_PYTHON AND NOT WIN32)
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
    regex-lang-test.cc
    slice-test.cc
    stack-test.cc
    text-utils-test.cc
    text2token-test.cc
    transpose-test.cc
    unbind-test.cc
    utfcpp-test.cc
  )
  if(SHERPA_ONNX_ENABLE_TTS)
    list(APPEND sherpa_onnx_test_srcs
      cppjieba-test.cc
      piper-phonemize-test.cc
    )
  endif()

  if(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION)
    list(APPEND sherpa_onnx_test_srcs
      fast-clustering-test.cc
    )
  endif()

  list(APPEND sherpa_onnx_test_srcs
    speaker-embedding-manager-test.cc
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

set(srcs_to_check)
foreach(s IN LISTS sources)
  list(APPEND srcs_to_check ${CMAKE_CURRENT_LIST_DIR}/${s})
endforeach()

# For clang-tidy
add_custom_target(
  clang-tidy-check
  clang-tidy -p ${CMAKE_BINARY_DIR}/compile_commands.json --config-file ${PROJECT_SOURCE_DIR}/.clang-tidy ${srcs_to_check}
  DEPENDS ${sources})

add_custom_target(check DEPENDS clang-tidy-check)
