#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z $EXE ]; then
  EXE=./build/bin/sherpa-onnx-offline-denoiser
fi

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run gtcrn"
log "------------------------------------------------------------"
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav

$EXE \
  --debug=1 \
  --speech-denoiser-gtcrn-model=./gtcrn_simple.onnx \
  --input-wav=./speech_with_noise.wav \
  --output-wav=./enhanced_speech_16k.wav

rm ./gtcrn_simple.onnx

log "------------------------------------------------------------"
log "Run voxlock"
log "------------------------------------------------------------"
voxlock_url=https://github.com/arbazsiddiqui/voxlock/releases/download/sherpa-onnx-models
download $voxlock_url/voxlock_enroll.onnx
download $voxlock_url/voxlock_chunk_int8.onnx
download $voxlock_url/voxlock_test_call.wav
download $voxlock_url/voxlock_test_voice_sample.wav

$EXE \
  --debug=1 \
  --speech-denoiser-voxlock-enroll-model=./voxlock_enroll.onnx \
  --speech-denoiser-voxlock-chunk-model=./voxlock_chunk_int8.onnx \
  --input-wav=./voxlock_test_call.wav \
  --output-wav=./voxlock_enhanced_speech_16k.wav

$EXE \
  --speech-denoiser-voxlock-enroll-model=./voxlock_enroll.onnx \
  --speech-denoiser-voxlock-chunk-model=./voxlock_chunk_int8.onnx \
  --speech-denoiser-voxlock-voice-sample=./voxlock_test_voice_sample.wav \
  --input-wav=./voxlock_test_call.wav \
  --output-wav=./voxlock_enhanced_speech_with_voice_sample_16k.wav

ONLINE_EXE=${EXE/offline-denoiser/online-denoiser}
which $ONLINE_EXE

$ONLINE_EXE \
  --speech-denoiser-voxlock-enroll-model=./voxlock_enroll.onnx \
  --speech-denoiser-voxlock-chunk-model=./voxlock_chunk_int8.onnx \
  --chunk-duration-ms=80 \
  --input-wav=./voxlock_test_call.wav \
  --output-wav=./voxlock_streaming_enhanced_speech_16k.wav

rm ./voxlock_enroll.onnx ./voxlock_chunk_int8.onnx
