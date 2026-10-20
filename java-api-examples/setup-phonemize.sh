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

PIPER_PHONEMIZE_VERSION=v1.5.9
PIPER_BASE_URL=https://jitpack.io/com/github/csukuangfj/piper-phonemize

# The native jar must match the JVM's architecture, not the shell's: Git
# Bash on Windows ARM reports x86_64 while the JVM is aarch64. This mirrors
# LibraryUtils.getOsArch() so the jar's internal resource path is found.
jvm_props=$(java -XshowSettings:properties -version 2>&1)
jvm_os=$(printf '%s\n' "$jvm_props" | sed -n 's/.*os\.name = //p' | head -1 |
  tr '[:upper:]' '[:lower:]')
jvm_arch=$(printf '%s\n' "$jvm_props" | sed -n 's/.*os\.arch = //p' | head -1 |
  tr '[:upper:]' '[:lower:]')

case "$jvm_os" in
  *mac* | *darwin*) piper_os=osx ;;
  *win*) piper_os=win ;;
  *nux*) piper_os=linux ;;
  *)
    echo "Unsupported OS: $jvm_os" >&2
    exit 1
    ;;
esac

case "$jvm_arch" in
  amd64* | x86_64*) piper_arch=x64 ;;
  aarch64* | arm64*)
    if [ "$piper_os" = "win" ]; then piper_arch=arm64; else piper_arch=aarch64; fi
    ;;
  x86*) piper_arch=x86 ;;
  arm*) piper_arch=arm ;;
  *)
    echo "Unsupported architecture: $jvm_arch" >&2
    exit 1
    ;;
esac

piper_plat="$piper_os-$piper_arch"

piper_jvm_jar=piper-phonemize-jvm-$PIPER_PHONEMIZE_VERSION.jar
piper_native_jar=piper-phonemize-native-lib-$piper_plat-$PIPER_PHONEMIZE_VERSION.jar

if [ ! -f ./$piper_jvm_jar ]; then
  curl -fSL -O $PIPER_BASE_URL/piper-phonemize-jvm/$PIPER_PHONEMIZE_VERSION/$piper_jvm_jar
fi

if [ ! -f ./$piper_native_jar ]; then
  curl -fSL -O \
    $PIPER_BASE_URL/piper-phonemize-native-lib-$piper_plat/$PIPER_PHONEMIZE_VERSION/$piper_native_jar
fi

case "$jvm_os" in
  *win*) piper_sep=';' ;;
  *) piper_sep=':' ;;
esac

# The sherpa-onnx jar name contains a version glob; expand it here.
sherpa_jar=$(ls ../sherpa-onnx/java-api/target/sherpa-onnx-jvm-*.jar)

# Complete classpath for the phonemize examples: sherpa-onnx + piper-phonemize
export PHONEMIZE_CLASSPATH="$sherpa_jar$piper_sep$piper_jvm_jar$piper_sep$piper_native_jar"
