#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "CXX_STREAMING_ZIPFORMER_EXE is $CXX_STREAMING_ZIPFORMER_EXE"
echo "CXX_WHISPER_EXE is $CXX_WHISPER_EXE"
echo "CXX_SENSE_VOICE_EXE is $CXX_SENSE_VOICE_EXE"
echo "PATH: $PATH"

log "------------------------------------------------------------"
log "Test streaming zipformer CXX API"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
$CXX_STREAMING_ZIPFORMER_EXE
rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

log "------------------------------------------------------------"
log "Test Whisper CXX API"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2
$CXX_WHISPER_EXE
rm -rf sherpa-onnx-whisper-tiny.en

log "------------------------------------------------------------"
log "Test SenseVoice CXX API"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

$CXX_SENSE_VOICE_EXE
rm -rf sherpa-onnx-sense-voice-*
