add_executable(online-fbank-test online-fbank-test.cc)
target_link_libraries(online-fbank-test kaldi-native-fbank-core)

include_directories( 
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/session/
    ${ONNXRUNTIME_ROOTDIR}/include/onnxruntime/core/providers/tensorrt/
)

include_directories(
    ${KALDINATIVEIO}
)
add_executable(sherpa-onnx main.cpp)
target_link_libraries(sherpa-onnx onnxruntime kaldi-native-fbank-core kaldi_native_io_core)
