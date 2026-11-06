#!/usr/bin/env bash
# Copyright      2026  danielxxomg

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

# Redux is a ternary v3: v3's .nemo supplies the architecture, Redux's
# safetensors supply the packed weights, and ternary.json specifies the packing.
curl -SL -O https://huggingface.co/moondream/parakeet-redux/resolve/main/model.safetensors
curl -SL -O https://huggingface.co/moondream/parakeet-redux/resolve/main/tokenizer.json
curl -SL -O https://huggingface.co/moondream/parakeet-redux/resolve/main/ternary.json
curl -SL -O https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/parakeet-tdt-0.6b-v3.nemo

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/en.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/de.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/fr.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/es.wav

ls -lh

pip install \
  nemo_toolkit['asr'] \
  "numpy<2" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx==1.17.0 \
  onnxruntime==1.17.1 \
  safetensors \
  soundfile

python3 ./export_onnx.py
ls -lh *.onnx

for w in en.wav de.wav fr.wav es.wav; do
  echo "---fp32----"
  python3 ./test_onnx.py \
    --encoder ./encoder.onnx \
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
done
