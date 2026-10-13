#!/usr/bin/env bash
# Common setup for the piper-phonemize based TTS examples.
# Source this file after ./setup.sh in each run-*-phonemize.sh script:
#   source ./setup.sh
#   source ./setup-phonemize.sh
#
# It downloads the piper-phonemize JVM jar and the native jar for the current
# platform from JitPack and sets PIPER_CLASSPATH for you. The espeak-ng-data
# used by the phonemizer is embedded in those jars; it only feeds piper
# phonemize, not sherpa-onnx.

PIPER_PHONEMIZE_VERSION=v1.5.4
PIPER_BASE_URL=https://jitpack.io/com/github/csukuangfj/piper-phonemize

os=$(uname -s)
arch=$(uname -m)
case "$os-$arch" in
  Darwin-arm64) piper_plat=osx-aarch64 ;;
  Darwin-x86_64) piper_plat=osx-x64 ;;
  Linux-x86_64) piper_plat=linux-x64 ;;
  Linux-aarch64) piper_plat=linux-aarch64 ;;
  MINGW*-x86_64 | MSYS*-x86_64 | CYGWIN*-x86_64) piper_plat=win-x64 ;;
  MINGW*-aarch64 | MSYS*-aarch64 | CYGWIN*-aarch64) piper_plat=win-arm64 ;;
  *)
    echo "Unsupported platform: $os-$arch" >&2
    exit 1
    ;;
esac

piper_jvm_jar=piper-phonemize-jvm-$PIPER_PHONEMIZE_VERSION.jar
piper_native_jar=piper-phonemize-native-lib-$piper_plat-$PIPER_PHONEMIZE_VERSION.jar

if [ ! -f ./$piper_jvm_jar ]; then
  curl -SL -O $PIPER_BASE_URL/piper-phonemize-jvm/$PIPER_PHONEMIZE_VERSION/$piper_jvm_jar
fi

if [ ! -f ./$piper_native_jar ]; then
  curl -SL -O \
    $PIPER_BASE_URL/piper-phonemize-native-lib-$piper_plat/$PIPER_PHONEMIZE_VERSION/$piper_native_jar
fi

case "$os" in
  MINGW* | MSYS* | CYGWIN*) piper_sep=';' ;;
  *) piper_sep=':' ;;
esac

# The sherpa-onnx jar name contains a version glob; expand it here.
sherpa_jar=$(ls ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar)

# Complete classpath for the phonemize examples: sherpa-onnx + piper-phonemize
export PHONEMIZE_CLASSPATH="$sherpa_jar$piper_sep$piper_jvm_jar$piper_sep$piper_native_jar"
