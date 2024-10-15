#!/usr/bin/env bash

set -ex

function install() {
  pip install torch==2.3.1+cpu torchaudio==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

  pushd /tmp
  git clone https://github.com/myshell-ai/MeloTTS
  cd MeloTTS
  pip install -r ./requirements.txt

  pip install soundfile onnx==1.15.0 onnxruntime==1.16.3

  python3 -m unidic download
  popd
}

install

export PYTHONPATH=/tmp/MeloTTS:$PYTHONPATH

echo "pwd: $PWD"

./export-onnx.py

ls -lh

./show-info.py

head lexicon.txt
echo "---"
tail lexicon.txt
echo "---"
head tokens.txt
echo "---"
tail tokens.txt

./test.py

ls -lh
