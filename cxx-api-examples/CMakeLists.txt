include_directories(${PROJECT_SOURCE_DIR})

add_executable(streaming-zipformer-cxx-api ./streaming-zipformer-cxx-api.cc)
target_link_libraries(streaming-zipformer-cxx-api sherpa-onnx-cxx-api)

add_executable(streaming-zipformer-with-hr-cxx-api ./streaming-zipformer-with-hr-cxx-api.cc)
target_link_libraries(streaming-zipformer-with-hr-cxx-api sherpa-onnx-cxx-api)

add_executable(speech-enhancement-gtcrn-cxx-api ./speech-enhancement-gtcrn-cxx-api.cc)
target_link_libraries(speech-enhancement-gtcrn-cxx-api sherpa-onnx-cxx-api)

add_executable(kws-cxx-api ./kws-cxx-api.cc)
target_link_libraries(kws-cxx-api sherpa-onnx-cxx-api)

add_executable(streaming-zipformer-rtf-cxx-api ./streaming-zipformer-rtf-cxx-api.cc)
target_link_libraries(streaming-zipformer-rtf-cxx-api sherpa-onnx-cxx-api)

add_executable(streaming-t-one-ctc-cxx-api   streaming-t-one-ctc-cxx-api.cc)
target_link_libraries(streaming-t-one-ctc-cxx-api sherpa-onnx-cxx-api)

add_executable(whisper-cxx-api ./whisper-cxx-api.cc)
target_link_libraries(whisper-cxx-api sherpa-onnx-cxx-api)

add_executable(fire-red-asr-cxx-api ./fire-red-asr-cxx-api.cc)
target_link_libraries(fire-red-asr-cxx-api sherpa-onnx-cxx-api)

add_executable(moonshine-cxx-api ./moonshine-cxx-api.cc)
target_link_libraries(moonshine-cxx-api sherpa-onnx-cxx-api)

add_executable(sense-voice-cxx-api ./sense-voice-cxx-api.cc)
target_link_libraries(sense-voice-cxx-api sherpa-onnx-cxx-api)

add_executable(wenet-ctc-cxx-api ./wenet-ctc-cxx-api.cc)
target_link_libraries(wenet-ctc-cxx-api sherpa-onnx-cxx-api)

add_executable(nemo-canary-cxx-api ./nemo-canary-cxx-api.cc)
target_link_libraries(nemo-canary-cxx-api sherpa-onnx-cxx-api)

add_executable(punctuation-cxx-api ./punctuation-cxx-api.cc)
target_link_libraries(punctuation-cxx-api sherpa-onnx-cxx-api)

if(SHERPA_ONNX_ENABLE_PORTAUDIO)
  add_executable(sense-voice-simulate-streaming-microphone-cxx-api
    ./sense-voice-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(sense-voice-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(wenet-ctc-simulate-streaming-microphone-cxx-api
    ./wenet-ctc-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(wenet-ctc-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(parakeet-tdt-simulate-streaming-microphone-cxx-api
    ./parakeet-tdt-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(parakeet-tdt-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(parakeet-tdt-ctc-simulate-streaming-microphone-cxx-api
    ./parakeet-tdt-ctc-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(parakeet-tdt-ctc-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(zipformer-ctc-simulate-streaming-microphone-cxx-api
    ./zipformer-ctc-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(zipformer-ctc-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(zipformer-transducer-simulate-streaming-microphone-cxx-api
    ./zipformer-transducer-simulate-streaming-microphone-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/microphone.cc
  )
  target_link_libraries(zipformer-transducer-simulate-streaming-microphone-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )
endif()

if(SHERPA_ONNX_HAS_ALSA)
  add_executable(sense-voice-simulate-streaming-alsa-cxx-api
    ./sense-voice-simulate-streaming-alsa-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/alsa.cc
  )
  target_link_libraries(sense-voice-simulate-streaming-alsa-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  add_executable(zipformer-ctc-simulate-streaming-alsa-cxx-api
    ./zipformer-ctc-simulate-streaming-alsa-cxx-api.cc
    ${CMAKE_CURRENT_LIST_DIR}/../sherpa-onnx/csrc/alsa.cc
  )
  target_link_libraries(zipformer-ctc-simulate-streaming-alsa-cxx-api
    sherpa-onnx-cxx-api
    portaudio_static
  )

  if(DEFINED ENV{SHERPA_ONNX_ALSA_LIB_DIR})
    target_link_libraries(sense-voice-simulate-streaming-alsa-cxx-api -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
    target_link_libraries(zipformer-ctc-simulate-streaming-alsa-cxx-api -L$ENV{SHERPA_ONNX_ALSA_LIB_DIR} -lasound)
  else()
    target_link_libraries(sense-voice-simulate-streaming-alsa-cxx-api asound)
    target_link_libraries(zipformer-ctc-simulate-streaming-alsa-cxx-api asound)
  endif()
endif()

add_executable(sense-voice-with-hr-cxx-api ./sense-voice-with-hr-cxx-api.cc)
target_link_libraries(sense-voice-with-hr-cxx-api sherpa-onnx-cxx-api)

add_executable(dolphin-ctc-cxx-api ./dolphin-ctc-cxx-api.cc)
target_link_libraries(dolphin-ctc-cxx-api sherpa-onnx-cxx-api)

add_executable(vad-cxx-api ./vad-cxx-api.cc)
target_link_libraries(vad-cxx-api sherpa-onnx-cxx-api)

if(SHERPA_ONNX_ENABLE_TTS)
  add_executable(matcha-tts-zh-cxx-api ./matcha-tts-zh-cxx-api.cc)
  target_link_libraries(matcha-tts-zh-cxx-api sherpa-onnx-cxx-api)

  add_executable(matcha-tts-en-cxx-api ./matcha-tts-en-cxx-api.cc)
  target_link_libraries(matcha-tts-en-cxx-api sherpa-onnx-cxx-api)

  add_executable(kokoro-tts-en-cxx-api ./kokoro-tts-en-cxx-api.cc)
  target_link_libraries(kokoro-tts-en-cxx-api sherpa-onnx-cxx-api)

  add_executable(kitten-tts-en-cxx-api ./kitten-tts-en-cxx-api.cc)
  target_link_libraries(kitten-tts-en-cxx-api sherpa-onnx-cxx-api)

  add_executable(kokoro-tts-zh-en-cxx-api ./kokoro-tts-zh-en-cxx-api.cc)
  target_link_libraries(kokoro-tts-zh-en-cxx-api sherpa-onnx-cxx-api)
endif()
