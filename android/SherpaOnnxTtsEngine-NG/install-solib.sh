#!/bin/bash

curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.9.10/sherpa-onnx-v1.9.10-android.tar.bz2 -o android.tar.bz2
tar -jxvf android.tar.bz2
rm -rf android.tar.bz2

echo "Move jniLibs to app/src/main/"
rm -rf app/src/main/jniLibs
mv jniLibs app/src/main/
