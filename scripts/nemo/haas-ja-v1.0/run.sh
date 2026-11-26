#!/usr/bin/env bash
# Copyright (c)  2026  Anthony Mercer

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

curl -SL -O https://huggingface.co/third-intelligence/Haas-ja-v1.0/resolve/main/Haas-ja-v1.0.nemo

curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2-ja-en/resolve/main/test_wavs/test_ja_1.wav
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2-ja-en/resolve/main/test_wavs/test_ja_2.wav

ls -lh

# haas-asr provides the model's decoder class and pins NeMo and
# flash-linear-attention. The export does not run the flash-linear-attention
# kernels, so it works on a CPU.
# torch 2.14.0 brings triton 3.8.0. flash-linear-attention 0.5.2 fails to
# import with triton 3.9.0, which comes with torch 2.15.0.
pip install \
  torch==2.14.0 \
  "haas-asr @ git+https://github.com/third-intelligence/Haas-ASR.git@acc614f68769dcbfae9dba96b48439c4a66295af" \
  ipython \
  kaldi-native-fbank \
  librosa \
  onnx==1.23.2 \
  onnxruntime==1.31.1 \
  soundfile

python3 ./export_onnx.py
ls -lh *.onnx

for w in test_ja_1.wav test_ja_2.wav; do
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
