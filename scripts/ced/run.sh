#!/usr/bin/env bash
#
# Copyright      2024  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

function install_dependencies() {
  pip install -qq torch==2.1.0+cpu torchaudio==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
  pip install -qq onnx onnxruntime==1.17.1

  pip install -r ./requirements.txt
}

git clone https://github.com/RicherMans/CED
pushd CED

install_dependencies

models=(
tiny
mini
small
base
)

for m in ${models[@]}; do
  python3 ./export_onnx.py -m ced_$m
done

ls -lh *.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
src=sherpa-onnx-zipformer-small-audio-tagging-2024-04-15

cat >README.md <<EOF
# Introduction

Models in this repo are converted from
https://github.com/RicherMans/CED
EOF

for m in ${models[@]}; do
  d=sherpa-onnx-ced-$m-audio-tagging-2024-04-19

  mkdir -p $d

  cp -v README.md $d
  cp -v $src/class_labels_indices.csv $d
  cp -a $src/test_wavs $d
  cp -v ced_$m.onnx $d/model.onnx
  cp -v ced_$m.int8.onnx $d/model.int8.onnx
  echo "----------$m----------"
  ls -lh $d
  echo "----------------------"
  tar cjvf $d.tar.bz2 $d
  mv $d.tar.bz2 ../../..
  mv $d ../../../
done

rm -rf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15

cd ../../..

ls -lh *.tar.bz2
echo "======="
ls -lh
