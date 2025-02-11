#!/usr/bin/env bash
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

function install_nemo() {
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py

  pip install torch==2.4.0 torchaudio==2.4.0 -f https://download.pytorch.org/whl/torch_stable.html

  pip install -qq wget text-unidecode matplotlib>=3.3.2 onnx onnxruntime pybind11 Cython einops kaldi-native-fbank soundfile librosa
  pip install -qq ipython

  # sudo apt-get install -q -y sox libsndfile1 ffmpeg python3-pip ipython

  BRANCH='main'
  python3 -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]

  pip install numpy==1.26.4
}

function download_files() {
  # curl -SL -O https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/ctc_model_weights.ckpt
  # curl -SL -O https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/ctc_model_config.yaml
  # curl -SL -O https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/example.wav
  # curl -SL -O https://n-ws-q0bez.s3pd12.sbercloud.ru/b-ws-q0bez-jpv/GigaAM/long_example.wav
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/ctc/ctc_model_weights.ckpt
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/ctc/ctc_model_config.yaml
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/example.wav
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/long_example.wav
  curl -SL -O https://huggingface.co/csukuangfj/tmp-files/resolve/main/GigaAM/GigaAM%20License_NC.pdf
}

install_nemo
download_files

python3 ./export-onnx-ctc.py
ls -lh
python3 ./test-onnx-ctc.py
