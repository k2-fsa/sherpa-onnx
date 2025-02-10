#!/usr/bin/env bash


set -ex
mkdir -p female male

if [ ! -f female/model.onnx ]; then
  curl -SL --output female/model.onnx https://huggingface.co/mah92/Khadijah-FA_EN-Matcha-TTS-Model/resolve/main/matcha-fa-en-khadijah-22050-5.onnx
fi

if [ ! -f female/tokens.txt ]; then
  curl -SL --output female/tokens.txt https://huggingface.co/mah92/Khadijah-FA_EN-Matcha-TTS-Model/resolve/main/tokens_sherpa_with_fa.txt
fi

if [ ! -f male/model.onnx ]; then
  curl -SL --output male/model.onnx https://huggingface.co/mah92/Musa-FA_EN-Matcha-TTS-Model/resolve/main/matcha-fa-en-musa-22050-5.onnx
fi

if [ ! -f male/tokens.txt ]; then
  curl -SL --output male/tokens.txt https://huggingface.co/mah92/Musa-FA_EN-Matcha-TTS-Model/resolve/main/tokens_sherpa_with_fa.txt
fi

if [ ! -f hifigan_v2.onnx ]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx
fi

if [ ! -f .add-meta-data.done ]; then
  python3 ./add_meta_data.py
  touch .add-meta-data.done
fi

python3 ./test.py \
  --am ./female/model.onnx \
  --vocoder ./hifigan_v2.onnx \
  --tokens ./female/tokens.txt \
  --text "This is a test. این یک نمونه ی تست فارسی است." \
  --out-wav "./female-en-fa.wav"

python3 ./test.py \
  --am ./male/model.onnx \
  --vocoder ./hifigan_v2.onnx \
  --tokens ./male/tokens.txt \
  --text "This is a test. این یک نمونه ی تست فارسی است." \
  --out-wav "./male-en-fa.wav"
