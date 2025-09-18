#!/usr/bin/env bash
set -ex

export PATH=/Users/fangjun/software/command-line-tools/bin:$PATH

cp -v ../../CHANGELOG.md ./sherpa_onnx

hvigorw clean --no-daemon
hvigorw --mode module -p product=default -p module=sherpa_onnx@default assembleHar --analyze=normal --parallel --incremental --no-daemon

ohpm publish ./sherpa_onnx/build/default/outputs/default/sherpa_onnx.har
