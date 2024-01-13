#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

d=/tmp/sr-models
mkdir -p $d

pushd $d
log "Download test waves"
git clone https://github.com/csukuangfj/sr-data
popd

log "Download wespeaker models"
model_dir=$d/wespeaker
mkdir -p $model_dir
pushd $model_dir
models=(
wespeaker_en_voxceleb_CAM++.onnx
wespeaker_en_voxceleb_CAM++_LM.onnx
wespeaker_en_voxceleb_resnet152_LM.onnx
wespeaker_en_voxceleb_resnet221_LM.onnx
wespeaker_en_voxceleb_resnet293_LM.onnx
wespeaker_en_voxceleb_resnet34.onnx
wespeaker_en_voxceleb_resnet34_LM.onnx
wespeaker_zh_cnceleb_resnet34.onnx
wespeaker_zh_cnceleb_resnet34_LM.onnx
)
for m in ${models[@]}; do
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$m
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_CAM++_LM.onnx
done
ls -lh
popd

log "Download 3d-speaker models"
model_dir=$d/3dspeaker
mkdir -p $model_dir
pushd $model_dir
models=(
3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx
3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx
3dspeaker_speech_eres2net_base_200k_sv_zh-cn_16k-common.onnx
3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx
3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx
3dspeaker_speech_eres2net_sv_zh-cn_16k-common.onnx
)
for m in ${models[@]}; do
  wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/$m
done
ls -lh
popd


python3 sherpa-onnx/python/tests/test_speaker_recognition.py --verbose
