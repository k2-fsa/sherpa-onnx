#!/usr/bin/env bash

set -ex

function install_gigaam() {
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py

  BRANCH='main'
  python3 -m pip install git+https://github.com/salute-developers/GigaAM.git@$BRANCH#egg=gigaam

  python3 -m pip install -qq kaldi-native-fbank
}

function download_files() {
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/example.wav
  curl -SL -O https://github.com/salute-developers/GigaAM/blob/main/LICENSE
}

install_gigaam
download_files

python3 ./export-onnx-ctc-v2.py
ls -lh
python3 ./test-onnx-ctc.py
