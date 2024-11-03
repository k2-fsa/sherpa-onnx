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

mkdir zh_en
mv -v *.onnx zh_en/
mv -v lexicon.txt zh_en
mv -v tokens.txt zh_en
cp -v README.md zh_en

ls -lh
echo "---"
ls -lh zh_en

./export-onnx-en.py

mkdir en
mv -v *.onnx en/
mv -v lexicon.txt en
mv -v tokens.txt en
cp -v README.md en

ls -lh en

ls -lh
