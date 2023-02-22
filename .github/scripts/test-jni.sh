#!/usr/bin/env bash

set -e

mkdir -p build
cd build

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

cd ..

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

cd .github/scripts/

git lfs install
git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21

kotlinc-jvm -include-runtime -d main.jar Main.kt WaveReader.kt SherpaOnnx.kt AssetManager.kt

ls -lh main.jar

java -Djava.library.path=../../build/lib -jar main.jar
