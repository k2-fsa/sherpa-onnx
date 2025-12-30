#!/usr/bin/env bash
set -ex

nm -g ../../build/lib/libsherpa-onnx-jni.dylib | awk '$2=="T" && $3 ~ /^_Java_com_k2fsa/ {print $3}' | sort  > ./sherpa-onnx-symbols.exp

