#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex


# Please see https://github.com/TRvlvr/model_repo/releases/tag/all_public_uvr_models
models=(
UVR-MDX-NET-Inst_1.onnx
UVR-MDX-NET-Inst_2.onnx
UVR-MDX-NET-Inst_3.onnx
UVR-MDX-NET-Inst_HQ_1.onnx
UVR-MDX-NET-Inst_HQ_2.onnx
UVR-MDX-NET-Inst_HQ_3.onnx
UVR-MDX-NET-Inst_HQ_4.onnx
UVR-MDX-NET-Inst_HQ_5.onnx
UVR-MDX-NET-Inst_Main.onnx
UVR-MDX-NET-Voc_FT.onnx
UVR-MDX-NET_Crowd_HQ_1.onnx
UVR_MDXNET_1_9703.onnx
UVR_MDXNET_2_9682.onnx
UVR_MDXNET_3_9662.onnx
UVR_MDXNET_9482.onnx
UVR_MDXNET_KARA.onnx
UVR_MDXNET_KARA_2.onnx
UVR_MDXNET_Main.onnx
)

mkdir -p models
for m in ${models[@]}; do
  if [ ! -f models/$m ]; then
    curl -SL --output models/$m https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/$m
  fi
done

ls -lh models

for m in ${models[@]}; do
  echo "----------$m----------"
  python3 ./add_meta_data_and_quantize.py --filename models/$m

  ls -lh models/
done

if [ -f ./audio_example.wav ]; then
  for m in ${models[@]}; do
    ./test.py  --model-filename ./models/$m --audio-filename ./audio_example.wav
    name=$(basename -s .onnx $m)
    mv -v vocals.mp3 ${name}_vocals.mp3
    mv -v non_vocals.mp3 ${name}_non_vocals.mp3
  done

  ls -lh *.mp3
fi
