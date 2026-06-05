#!/usr/bin/env bash


# We run this script inside a docker container using GitHub actions
# sherpa-onnx repo is mounted at /sherpa-onnx
# icefall repo is mounted at /icefall
set -ex

chunk_size=8
if [ $# -ge 1 ]; then
  chunk_size=$(($1/20))
fi

echo "pwd: $PWD"

echo "download models"

mkdir -p punct no-punct test_wavs

pushd test_wavs
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/0.wav
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/1.wav
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/2.wav
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/3.wav
popd

pushd punct
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/README.md
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/bpe_punc.model
mv bpe_punc.model bpe.model
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/tokens.txt
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/checkpoint/fintuned_with_punctuation.pt
ln -s fintuned_with_punctuation.pt epoch-99.pt
popd

pushd no-punct
curl -SL -O https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/README.md
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/checkpoint/pretrained.pt
ln -s pretrained.pt epoch-99.pt
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000/tokens.txt
curl -SL -O https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000/bpe.model
popd

dir=$PWD

pushd /icefall/egs/librispeech/ASR/

./zipformer/export-onnx-streaming.py \
  --use-int32-inputs 0 \
  --dynamic-batch 1 \
  --enable-int8-quantization 1 \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir $dir/punct \
  --tokens $dir/punct/tokens.txt \
  \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --causal 1 \
  --chunk-size $chunk_size \
  --left-context-frames 256

ls -lh $dir/punct

pushd $dir/punct
mv encoder-*256.onnx encoder.onnx
mv decoder-*256.onnx decoder.onnx
mv joiner-*256.onnx joiner.onnx

mv encoder-*256.int8.onnx encoder.int8.onnx
rm decoder-*256.int8.onnx
mv joiner-*256.int8.onnx joiner.int8.onnx
popd

ls -lh $dir/punct

echo "----"

./zipformer/export-onnx-streaming.py \
  --use-int32-inputs 0 \
  --dynamic-batch 1 \
  --enable-int8-quantization 1 \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  --exp-dir $dir/no-punct \
  --tokens $dir/no-punct/tokens.txt \
  \
  --num-encoder-layers 2,2,4,5,4,2 \
  --feedforward-dim 512,768,1536,2048,1536,768 \
  --encoder-dim 192,256,512,768,512,256 \
  --encoder-unmasked-dim 192,192,256,320,256,192 \
  --causal 1 \
  --chunk-size $chunk_size \
  --left-context-frames 256

ls -lh $dir/no-punct

pushd $dir/no-punct
mv encoder-*256.onnx encoder.onnx
mv decoder-*256.onnx decoder.onnx
mv joiner-*256.onnx joiner.onnx

mv encoder-*256.int8.onnx encoder.int8.onnx
rm decoder-*256.int8.onnx
mv joiner-*256.int8.onnx joiner.int8.onnx
popd

ls -lh $dir/no-punct

popd

echo "testing"

cp -v /sherpa-onnx/scripts/zipformer-transducer/x-asr/test_onnx_streaming.py punct/test_onnx.py
cp -v /sherpa-onnx/scripts/zipformer-transducer/x-asr/test_onnx_streaming.py no-punct/test_onnx.py

pushd punct
for w in 0 1 2 3; do
  for use_int8 in 0 1; do
    python3 ./test_onnx.py --use-int8 $use_int8 --wav ../test_wavs/$w.wav
  done
done
popd

pushd no-punct
for w in 0 1 2 3; do
  for use_int8 in 0 1; do
    python3 ./test_onnx.py --use-int8 $use_int8 --wav ../test_wavs/$w.wav
  done
done
popd
