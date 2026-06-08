#!/usr/bin/env bash


# We run this script inside a docker container using GitHub actions
# sherpa-onnx repo is mounted at /sherpa-onnx
# icefall repo is mounted at /icefall
set -ex

# Download a file with retries and size validation
# Usage: download_file URL [MIN_SIZE_KB]
# If the file is an HTML page or smaller than MIN_SIZE_KB, retry the download
download_file() {
  local url=$1
  local min_size_kb=${2:-10}  # default minimum 10KB
  local filename=$(basename "$url")
  local max_retries=5
  local retry_delay=5

  for i in $(seq 1 $max_retries); do
    echo "Download attempt $i/$max_retries: $url"
    curl -SL -O "$url" || true

    # Check if file exists
    if [ ! -f "$filename" ]; then
      echo "Error: File $filename not found after download"
      sleep $retry_delay
      continue
    fi

    # Check if file is HTML (error page)
    if file "$filename" | grep -q "HTML"; then
      echo "Error: Downloaded file is HTML, likely a redirect/error page"
      rm -f "$filename"
      sleep $retry_delay
      continue
    fi

    # Check file size
    local size_kb=$(du -k "$filename" | cut -f1)
    if [ "$size_kb" -lt "$min_size_kb" ]; then
      echo "Error: File $filename is too small (${size_kb}KB < ${min_size_kb}KB)"
      rm -f "$filename"
      sleep $retry_delay
      continue
    fi

    echo "Download successful: $filename (${size_kb}KB)"
    return 0
  done

  echo "Error: Failed to download $url after $max_retries attempts"
  return 1
}

chunk_size=8
if [ $# -ge 1 ]; then
  chunk_size=$(($1/20))
fi

echo "pwd: $PWD"

echo "download models"

mkdir -p punct no-punct test_wavs

pushd test_wavs
download_file https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/0.wav 1
download_file https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/1.wav 1
download_file https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/2.wav 1
download_file https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/3.wav 1
popd

pushd punct
download_file https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/README.md 1
download_file https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/bpe_punc.model 10
mv bpe_punc.model bpe.model
download_file https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000_with_punctuation/tokens.txt 10
download_file https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/checkpoint/fintuned_with_punctuation.pt 100
ln -s fintuned_with_punctuation.pt epoch-99.pt
echo "pwd"
ls -lh
file *.pt
popd

pushd no-punct
download_file https://modelscope.cn/models/csukuangfj/2026-06-03/resolve/master/README.md 1
download_file https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/checkpoint/pretrained.pt 100
ln -s pretrained.pt epoch-99.pt
download_file https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000/tokens.txt 10
download_file https://github.com/Gilgamesh-J/X-ASR/raw/refs/heads/main/X-ASR-zh-en/zipformer/data/lang_5000/bpe.model 10
echo "pwd"
ls -lh
file *.pt
popd

dir=$PWD

pushd /icefall/egs/librispeech/ASR/

./zipformer/export-onnx-streaming.py \
  --use-int32-inputs 1 \
  --dynamic-batch 0 \
  --enable-int8-quantization 0 \
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
popd

ls -lh $dir/punct

echo "----"

./zipformer/export-onnx-streaming.py \
  --use-int32-inputs 1 \
  --dynamic-batch 0 \
  --enable-int8-quantization 0 \
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
popd

ls -lh $dir/no-punct

popd

echo "testing"

cp -v /sherpa-onnx/scripts/zipformer-transducer/x-asr/test_onnx_streaming.py punct/test_onnx.py
cp -v /sherpa-onnx/scripts/zipformer-transducer/x-asr/test_onnx_streaming.py no-punct/test_onnx.py

pushd punct
for w in 0 1 2 3; do
  python3 ./test_onnx.py --use-int8 0 --wav ../test_wavs/$w.wav
done
popd

pushd no-punct
for w in 0 1 2 3; do
  python3 ./test_onnx.py --use-int8 0 --wav ../test_wavs/$w.wav
done
popd

rm -v punct/*.pt
rm -v no-punct/*.pt
