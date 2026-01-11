#!/usr/bin/env bash

set -ex

function install_gigaam() {
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py
  pip install torch==2.4.0 torchaudio==2.4.0 -f https://download.pytorch.org/whl/torch_stable.html
  pip install -qq wget text-unidecode "matplotlib>=3.3.2" onnx onnxruntime==1.17.1 pybind11 Cython einops kaldi-native-fbank soundfile librosa

  BRANCH='main'
  python3 -m pip install git+https://github.com/salute-developers/GigaAM.git@$BRANCH#egg=gigaam

  python3 -m pip install -qq kaldi-native-fbank
  pip install numpy==1.26.4
}

function download_files() {
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/example.wav
  curl -SL -O https://raw.githubusercontent.com/salute-developers/GigaAM/main/LICENSE
}

install_gigaam
download_files

python3 ./export-onnx-ctc-v3.py
ls -lh
python3 ./test-onnx-ctc.py
