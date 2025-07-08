#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

if [ -z $EXE ]; then
  EXE=./build/bin/sherpa-onnx-offline-source-separation
fi

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run spleeter"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
tar xvf sherpa-onnx-spleeter-2stems-fp16.tar.bz2
rm sherpa-onnx-spleeter-2stems-fp16.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav

$EXE \
  --spleeter-vocals=sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx \
  --spleeter-accompaniment=sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx \
  --num-threads=2 \
  --debug=1 \
  --input-wav=./qi-feng-le-zh.wav \
  --output-vocals-wav=spleeter_output_vocals.wav \
  --output-accompaniment-wav=spleeter_output_accompaniment.wav

rm -rf sherpa-onnx-spleeter-2stems-fp16

log "------------------------------------------------------------"
log "Run UVR"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx

$EXE \
  --debug=1 \
  --num-threads=2 \
  --uvr-model=./UVR-MDX-NET-Voc_FT.onnx \
  --input-wav=./qi-feng-le-zh.wav \
  --output-vocals-wav=uvr_output_vocals.wav \
  --output-accompaniment-wav=uvr_output_non_vocals.wav

rm ./UVR-MDX-NET-Voc_FT.onnx \

mkdir source-separation-wavs
mv qi-feng-le-zh.wav source-separation-wavs
mv spleeter_*.wav ./source-separation-wavs
mv uvr_*.wav ./source-separation-wavs
