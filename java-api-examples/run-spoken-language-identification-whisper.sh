#!/usr/bin/env bash

set -ex

if [[ ! -f ../build/lib/libsherpa-onnx-jni.dylib  && ! -f ../build/lib/libsherpa-onnx-jni.so ]]; then
  mkdir -p ../build
  pushd ../build
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
  popd
fi

if [ ! -f ../sherpa-onnx/java-api/build/sherpa-onnx.jar ]; then
  pushd ../sherpa-onnx/java-api
  make
  popd
fi

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

# Note that it needs a multilingual whisper model. so, for example, tiny works while tiny.en does not work
# https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
  tar xvf sherpa-onnx-whisper-tiny.tar.bz2
  rm sherpa-onnx-whisper-tiny.tar.bz2
fi

if [ ! -f ./spoken-language-identification-test-wavs/en-english.wav ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
  tar xvf spoken-language-identification-test-wavs.tar.bz2
  rm spoken-language-identification-test-wavs.tar.bz2
fi

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/build/sherpa-onnx.jar \
  ./SpokenLanguageIdentificationWhisper.java
