#!/usr/bin/env bash

set -e

function install_3d_speaker() {
  echo "Install 3D-Speaker"
  git clone https://github.com/modelscope/3D-Speaker
  pushd 3D-Speaker
  pip install -q -r ./requirements.txt
  pip install -q modelscope==1.14.0 onnx onnxruntime kaldi-native-fbank
  popd
}

function download_test_data() {
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_a_cn_16k.wav
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_b_cn_16k.wav
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker2_a_cn_16k.wav

  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_a_en_16k.wav
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_b_en_16k.wav
  wget -q https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker2_a_en_16k.wav
}

install_3d_speaker

download_test_data

export PYTHONPATH=$PWD/3D-Speaker:$PYTHONPATH
export PYTHONPATH=$PWD/3D-Speaker/speakerlab/bin:$PYTHONPATH

models=(
speech_campplus_sv_en_voxceleb_16k
speech_campplus_sv_zh-cn_16k-common
speech_campplus_sv_zh_en_16k-common_advanced
speech_eres2net_sv_en_voxceleb_16k
speech_eres2net_sv_zh-cn_16k-common
speech_eres2net_base_200k_sv_zh-cn_16k-common
speech_eres2net_base_sv_zh-cn_3dspeaker_16k
speech_eres2net_large_sv_zh-cn_3dspeaker_16k
speech_eres2netv2_sv_zh-cn_16k-common
)
for model in ${models[@]}; do
  echo "--------------------$model--------------------"
  python3 ./export-onnx.py --model $model

  python3 ./test-onnx.py \
    --model ${model}.onnx \
    --file1 ./speaker1_a_cn_16k.wav \
    --file2 ./speaker1_b_cn_16k.wav

  python3 ./test-onnx.py \
    --model ${model}.onnx \
    --file1 ./speaker1_a_cn_16k.wav \
    --file2 ./speaker2_a_cn_16k.wav

  python3 ./test-onnx.py \
    --model ${model}.onnx \
    --file1 ./speaker1_a_en_16k.wav \
    --file2 ./speaker1_b_en_16k.wav

  python3 ./test-onnx.py \
    --model ${model}.onnx \
    --file1 ./speaker1_a_en_16k.wav \
    --file2 ./speaker2_a_en_16k.wav

  mv ${model}.onnx 3dspeaker_${model}.onnx
done
