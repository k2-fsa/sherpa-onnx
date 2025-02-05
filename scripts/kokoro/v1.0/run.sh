#!/usr/bin/env bash
# Copyright    2025  Xiaomi Corp.        (authors: Fangjun Kuang)

set -ex

if [ ! -f kokoro.onnx ]; then
  # see https://github.com/taylorchu/kokoro-onnx/releases
  curl -SL -O https://github.com/taylorchu/kokoro-onnx/releases/download/v0.2.0/kokoro.onnx
fi

if [ ! -f config.json ]; then
  # see https://huggingface.co/hexgrad/Kokoro-82M/blob/main/config.json
  curl -SL -O https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/config.json
fi

# see https://huggingface.co/spaces/hexgrad/Kokoro-TTS/blob/main/app.py#L83
# and
# https://huggingface.co/hexgrad/Kokoro-82M/tree/main/voices
#
# af -> American female
# am -> American male
# bf -> British female
# bm -> British male
voices=(
af_alloy
af_aoede
af_bella
af_heart
af_jessica
af_kore
af_nicole
af_nova
af_river
af_sarah
af_sky
am_adam
am_echo
am_eric
am_fenrir
am_liam
am_michael
am_onyx
am_puck
am_santa
bf_alice
bf_emma
bf_isabella
bf_lily
bm_daniel
bm_fable
bm_george
bm_lewis
ef_dora
em_alex
ff_siwis
hf_alpha
hf_beta
hm_omega
hm_psi
if_sara
im_nicola
jf_alpha
jf_gongitsune
jf_nezumi
jf_tebukuro
jm_kumo
pf_dora
pm_alex
pm_santa
zf_xiaobei # 东北话
zf_xiaoni
zf_xiaoxiao
zf_xiaoyi
zm_yunjian
zm_yunxi
zm_yunxia
zm_yunyang
)

mkdir -p voices

for v in ${voices[@]}; do
  if [ ! -f voices/$v.pt ]; then
    curl -SL --output voices/$v.pt https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/$v.pt
  fi
done

if [ ! -f ./.add-meta-data.done ]; then
  python3 ./add_meta_data.py
  touch ./.add-meta-data.done
fi

if [ ! -f us_gold.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/us_gold.json
fi

if [ ! -f us_silver.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/us_silver.json
fi

if [ ! -f gb_gold.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/gb_gold.json
fi

if [ ! -f gb_silver.json ]; then
  curl -SL -O https://raw.githubusercontent.com/hexgrad/misaki/refs/heads/main/misaki/data/gb_silver.json
fi

if [ ! -f ./tokens.txt ]; then
  ./generate_tokens.py
fi

if [ ! -f ./lexicon.txt ]; then
  ./generate_lexicon.py
fi

if [ ! -f ./voices.bin ]; then
  ./generate_voices_bin.py
fi

./test.py
ls -lh
