#!/usr/bin/env bash

set -ex

# We run this script inside a docker container using GitHub actions
# sherpa-onnx repo is mounted at /sherpa-onnx
# icefall repo is mounted at /icefall

num_frames=1000

if [ $# -ge 1 ]; then
  num_frames="$1"
fi

echo "num_frames: $num_frames"

echo "download models"
# The original model is from
# https://huggingface.co/reazon-research/reazonspeech-k2-v2
# I have created a PyTorch checkpoint from it and put it in
# https://huggingface.co/csukuangfj/reazonspeech-k2-v2/tree/main/checkpoint

mkdir -p models
pushd models
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/checkpoint/model.pt
ln -s model.pt epoch-99.pt
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/tokens.txt

curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-11/resolve/master/test_onnx.py
chmod +x test_onnx.py

mkdir test_wavs
pushd test_wavs
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/1.wav
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/2.wav
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/3.wav
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/4.wav
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/5.wav
curl -SL -O https://huggingface.co/csukuangfj/reazonspeech-k2-v2/resolve/main/test_wavs/transcript.txt
popd

popd

dir=$PWD/models

pushd /icefall/egs/librispeech/ASR/

./zipformer/export-onnx.py \
  --enable-int8-quantization 0 \
  --max-len $num_frames \
  --keep-x-lens 0 \
  --use-int32-inputs 1 \
  --dynamic-axes 0 \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir $dir \
  --tokens $dir/tokens.txt \
  \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192

ls -lh $dir

popd

echo "test onnx files"

pushd $dir
./test_onnx.py --wav ./test_wavs/1.wav
./test_onnx.py --wav ./test_wavs/2.wav
./test_onnx.py --wav ./test_wavs/3.wav
./test_onnx.py --wav ./test_wavs/4.wav
./test_onnx.py --wav ./test_wavs/5.wav


popd

echo "copy onnx model files"

src=/sherpa-onnx/model-files
mkdir -p $src

mv -v $dir/encoder-epoch-99-avg-1.onnx $src/encoder.onnx
mv -v $dir/decoder-epoch-99-avg-1.onnx $src/decoder.onnx
mv -v $dir/joiner-epoch-99-avg-1.onnx $src/joiner.onnx

cp -v $dir/tokens.txt $src
cp -av $dir/test_wavs $src
rm -rf $dir/
