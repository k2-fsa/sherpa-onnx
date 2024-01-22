#!/usr/bin/env bash
#
# This scripts shows how to build JNI libs for sherpa-onnx
# Note: This scripts runs only on Linux and macOS, though sherpa-onnx
# supports building JNI libs for Windows.

set -e


cd ..
mkdir -p build
cd build

if [[ ! -f ../build/lib/libsherpa-onnx-jni.dylib  && ! -f ../build/lib/libsherpa-onnx-jni.so ]]; then
  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    ..

  make -j4
  ls -lh lib
fi

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

cd ../kotlin-api-examples

if [ ! -f ./3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx ]; then
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx
fi

if [ ! -f ./speaker1_a_cn_16k.wav ]; then
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_a_cn_16k.wav
fi

if [ ! -f ./speaker1_b_cn_16k.wav ]; then
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_b_cn_16k.wav
fi

if [ ! -f ./speaker2_a_cn_16k.wav ]; then
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker2_a_cn_16k.wav
fi

if [ ! -f ./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt ]; then
  git lfs install
  git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21
fi

if [ ! -d ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13 ]; then
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
fi

if [ ! -f ./vits-piper-en_US-amy-low/en_US-amy-low.onnx ]; then
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  tar xf vits-piper-en_US-amy-low.tar.bz2
  rm vits-piper-en_US-amy-low.tar.bz2
fi

kotlinc-jvm -include-runtime -d main.jar Main.kt WaveReader.kt SherpaOnnx.kt faked-asset-manager.kt Tts.kt Speaker.kt

ls -lh main.jar

java -Djava.library.path=../build/lib -jar main.jar

# For two-pass

if [ ! -f ./sherpa-onnx-streaming-zipformer-en-20M-2023-02-17/encoder-epoch-99-avg-1.int8.onnx ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
  tar xvf sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
  rm sherpa-onnx-streaming-zipformer-en-20M-2023-02-17.tar.bz2
fi

if [ ! -f ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx ]; then
  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
  rm sherpa-onnx-whisper-tiny.en.tar.bz2
fi

kotlinc-jvm -include-runtime -d 2pass.jar test-2pass.kt WaveReader.kt SherpaOnnx2Pass.kt faked-asset-manager.kt
ls -lh 2pass.jar
java -Djava.library.path=../build/lib -jar 2pass.jar
