#!/usr/bin/env bash
# Copyright      2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/de.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/fr.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/es.wav


pip install \
  nemo_toolkit['asr'] \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx==1.17.0 \
  onnxmltools==1.13.0 \
  onnxruntime==1.17.1 \
  soundfile

python3 ./export_onnx.py
ls -lh *.onnx

for w in en.wav de.wav fr.wav es.wav; do
  echo "---fp32----"
  python3 ./test_onnx.py \
    --encoder ./encoder.int8.onnx \
    --decoder ./decoder.onnx \
    --joiner ./joiner.onnx \
    --tokens ./tokens.txt \
    --wav $w

  echo "---int8----"
  python3 ./test_onnx.py \
    --encoder ./encoder.int8.onnx \
    --decoder ./decoder.int8.onnx \
    --joiner ./joiner.int8.onnx \
    --tokens ./tokens.txt \
    --wav $w

  echo "---fp16----"
  python3 ./test_onnx.py \
    --encoder ./encoder.fp16.onnx \
    --decoder ./decoder.fp16.onnx \
    --joiner ./joiner.fp16.onnx \
    --tokens ./tokens.txt \
    --wav $w
done
