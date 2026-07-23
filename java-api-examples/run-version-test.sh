#!/usr/bin/env bash

set -ex

source ./setup.sh

java \
  -Djava.library.path=$PWD/../build/lib \
  -cp ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar \
  ./VersionTest.java

