#!/usr/bin/env bash

if [ ! -d ./sherpa-onnx-nemo-ctc-en-conformer-medium ]; then
  GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/sherpa-onnx-nemo-ctc-en-conformer-medium
  cd sherpa-onnx-nemo-ctc-en-conformer-medium
  git lfs pull --include "*.onnx"
  cd ..
fi

dotnet run \
  --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
  --nemo-ctc=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
  --num-threads=1 \
  --files ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav
