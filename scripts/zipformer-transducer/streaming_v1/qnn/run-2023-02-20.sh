#!/usr/bin/env bash


# We run this script inside a docker container using GitHub actions
# sherpa-onnx repo is mounted at /sherpa-onnx
# icefall repo is mounted at /icefall

chunk_size=32

if [ $# -ge 1 ]; then
  chunk_size="$1"
fi

echo "chunk_size: $chunk_size"

echo "download models"
git clone https://huggingface.co/pfluo/k2fsa-zipformer-chinese-english-mixed
pushd k2fsa-zipformer-chinese-english-mixed/exp
git lfs pull
rm *.onnx
popd

dir=$PWD/k2fsa-zipformer-chinese-english-mixed

pushd /icefall/egs/librispeech/ASR/

./pruned_transducer_stateless7_streaming/export-onnx-zh.py \
  --exp-dir $dir/exp \
  --tokens $dir/data/lang_char_bpe/tokens.txt \
  --epoch 99 \
  --avg 1 \
  --use-averaged-model 0 \
  \
  --decode-chunk-len $chunk_size \
  --num-encoder-layers "2,4,3,2,4" \
  --feedforward-dims "1024,1024,1536,1536,1024" \
  --nhead "8,8,8,8,8" \
  --encoder-dims "384,384,384,384,384" \
  --attention-dims "192,192,192,192,192" \
  --encoder-unmasked-dims "256,256,256,256,256" \
  --zipformer-downsampling-factors "1,2,4,8,2" \
  --cnn-module-kernels "31,31,31,31,31" \
  --decoder-dim 512 \
  --joiner-dim 512 \
  --dynamic-batch 0 \
  --enable-int8-quantization 0 \
  --use-int32-inputs 1

ls -lh $dir/exp

echo "copy onnx model files"

src=/sherpa-onnx/model-files/$chunk_size
mkdir -p $src

mv -v $dir/exp/*.onnx $src/
cp -v $dir/data/lang_char_bpe/tokens.txt $src
rm -rf $dir
