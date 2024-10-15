#!/usr/bin/env bash
# Copyright    2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

function install_nemo() {
  curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
  python3 get-pip.py

  pip install torch==2.1.0 torchaudio==2.1.0 -f https://download.pytorch.org/whl/torch_stable.html

  pip install -qq wget text-unidecode matplotlib>=3.3.2 onnx onnxruntime pybind11 Cython einops kaldi-native-fbank soundfile
  pip install -qq ipython

  # sudo apt-get install -q -y sox libsndfile1 ffmpeg python3-pip ipython

  BRANCH='main'
  python3 -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]

  pip install numpy==1.26.4
}

install_nemo

model_list=(
speakerverification_speakernet
titanet_large
titanet_small
# ecapa_tdnn # causes errors, see https://github.com/NVIDIA/NeMo/issues/8168
)

for model in ${model_list[@]}; do
  python3 ./export-onnx.py --model $model
done

ls -lh

function download_test_data() {
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_a_en_16k.wav
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_b_en_16k.wav
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker2_a_en_16k.wav
}

download_test_data

for model in ${model_list[@]}; do
  python3 ./test-onnx.py \
    --model nemo_en_${model}.onnx \
    --file1 ./speaker1_a_en_16k.wav \
    --file2 ./speaker1_b_en_16k.wav

  python3 ./test-onnx.py \
    --model nemo_en_${model}.onnx \
    --file1 ./speaker1_a_en_16k.wav \
    --file2 ./speaker2_a_en_16k.wav
done
