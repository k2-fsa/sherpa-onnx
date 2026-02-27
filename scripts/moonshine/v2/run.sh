#!/usr/bin/env bash
# Copyright      2024-2026  Xiaomi Corp.        (authors: Fangjun Kuang)
set -ex

wget https://raw.githubusercontent.com/moonshine-ai/moonshine/refs/heads/main/LICENSE

d=$PWD/models
mkdir -p $d
echo "d: $d"

export MOONSHINE_VOICE_CACHE=$d

python3 -m moonshine_voice.download --language zh --model-arch 1
python3 -m moonshine_voice.download --language ar --model-arch 1
python3 -m moonshine_voice.download --language es --model-arch 1
python3 -m moonshine_voice.download --language en --model-arch 0
python3 -m moonshine_voice.download --language en --model-arch 1
python3 -m moonshine_voice.download --language ja --model-arch 0
python3 -m moonshine_voice.download --language ja --model-arch 1
python3 -m moonshine_voice.download --language ko --model-arch 0
python3 -m moonshine_voice.download --language vi --model-arch 1
python3 -m moonshine_voice.download --language uk --model-arch 1

sleep 2

ls -lh models/download.moonshine.ai/model/*/*/*

sleep 2

names=(
  base-ar
  base-en
  base-es
  base-ja
  base-uk
  base-vi
  base-zh
  tiny-en
  tiny-ja
  tiny-ko
)

for name in ${names[@]}; do
  mv -v models/download.moonshine.ai/model/$name/quantized/$name/* .
  python3 ./generate_tokens.py
  rm tokenizer.bin
  d=sherpa-onnx-moonshine-$name-quantized-2026-02-27
  mkdir -p $d
  mv *ort $d/
  cp LICENSE $d
  mv tokens.txt $d

  lang="${name##*-}"
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$lang.wav
  mv $lang.wav 0.wav

  mkdir $d/test_wavs
  mv 0.wav $d/test_wavs

  tar cjfv $d.tar.bz2 $d
  ls -lh
  mv -v $d ../../../
  mv -v $d.tar.bz2 ../../../
done

#
# models/download.moonshine.ai/model/base-ar/quantized/base-ar:
# total 135M
# -rw-r--r-- 1 root root 105M Feb 27 07:42 decoder_model_merged.ort
# -rw-r--r-- 1 root root  30M Feb 27 07:42 encoder_model.ort
# -rw-r--r-- 1 root root 245K Feb 27 07:42 tokenizer.bin
#
# models/download.moonshine.ai/model/base-en/quantized/base-en:
# total 135M
# -rw-r--r-- 1 root root 105M Feb 27 07:43 decoder_model_merged.ort
# -rw-r--r-- 1 root root  30M Feb 27 07:43 encoder_model.ort
# -rw-r--r-- 1 root root 245K Feb 27 07:43 tokenizer.bin
#
# models/download.moonshine.ai/model/base-es/quantized/base-es:
# total 62M
# -rw-r--r-- 1 root root  42M Feb 27 07:42 decoder_model_merged.ort
# -rw-r--r-- 1 root root  20M Feb 27 07:42 encoder_model.ort
# -rw-r--r-- 1 root root 236K Feb 27 07:42 tokenizer.bin
#
# models/download.moonshine.ai/model/base-ja/quantized/base-ja:
# total 135M
# -rw-r--r-- 1 root root 105M Feb 27 07:44 decoder_model_merged.ort
# -rw-r--r-- 1 root root  30M Feb 27 07:44 encoder_model.ort
# -rw-r--r-- 1 root root 245K Feb 27 07:44 tokenizer.bin
#
# models/download.moonshine.ai/model/base-zh/quantized/base-zh:
# total 135M
# -rw-r--r-- 1 root root 105M Feb 27 07:39 decoder_model_merged.ort
# -rw-r--r-- 1 root root  30M Feb 27 07:39 encoder_model.ort
# -rw-r--r-- 1 root root 245K Feb 27 07:39 tokenizer.bin
#
# models/download.moonshine.ai/model/tiny-en/quantized/tiny-en:
# total 42M
# -rw-r--r-- 1 root root  30M Feb 27 07:44 decoder_model_merged.ort
# -rw-r--r-- 1 root root  13M Feb 27 07:44 encoder_model.ort
# -rw-r--r-- 1 root root 245K Feb 27 07:44 tokenizer.bin
#
# models/download.moonshine.ai/model/tiny-ja/quantized/tiny-ja:
# total 69M
# -rw-r--r-- 1 root root  56M Feb 27 07:44 decoder_model_merged.ort
# -rw-r--r-- 1 root root  13M Feb 27 07:44 encoder_model.ort
# -rw-r--r-- 1 root root 245K Feb 27 07:44 tokenizer.bin
