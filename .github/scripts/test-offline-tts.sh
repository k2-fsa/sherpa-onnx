#!/usr/bin/env bash

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export GIT_CLONE_PROTECTION_ACTIVE=false

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

# test waves are saved in ./tts
mkdir ./tts

log "------------------------------------------------------------"
log "kokoro-en-v0_19"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

# mapping of sid to voice name
# 0->af, 1->af_bella, 2->af_nicole, 3->af_sarah, 4->af_sky, 5->am_adam
# 6->am_michael, 7->bf_emma, 8->bf_isabella, 9->bm_george, 10->bm_lewis

for sid in $(seq 0 10); do
  $EXE \
    --debug=1 \
    --kokoro-model=./kokoro-en-v0_19/model.onnx \
    --kokoro-voices=./kokoro-en-v0_19/voices.bin \
    --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
    --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
    --num-threads=2 \
    --sid=$sid \
    --output-filename="./tts/kokoro-$sid.wav" \
    "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be  a statesman, a businessman, an official, or a scholar."
done
rm -rf kokoro-en-v0_19

log "------------------------------------------------------------"
log "matcha-icefall-en_US-ljspeech"
log "------------------------------------------------------------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

$EXE \
  --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --matcha-vocoder=./hifigan_v2.onnx \
  --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
  --matcha-data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
  --num-threads=2 \
  --output-filename=./tts/matcha-ljspeech-1.wav \
  --debug=1 \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

rm hifigan_v2.onnx
rm -rf matcha-icefall-en_US-ljspeech

log "------------------------------------------------------------"
log "matcha-icefall-zh-baker"
log "------------------------------------------------------------"
curl -O -SL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

$EXE \
  --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
  --matcha-vocoder=./hifigan_v2.onnx \
  --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
  --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
  --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
  --num-threads=2 \
  --debug=1 \
  --output-filename=./tts/matcha-baker-zh-1.wav \
  '小米的使命是，始终坚持做"感动人心、价格厚道"的好产品，让全球每个人都能享受科技带来的美好生活'

$EXE \
  --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
  --matcha-vocoder=./hifigan_v2.onnx \
  --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
  --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
  --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
  --num-threads=2 \
  --debug=1 \
  --output-filename=./tts/matcha-baker-zh-2.wav \
  "当夜幕降临，星光点点，伴随着微风拂面，我在静谧中感受着时光的流转，思念如涟漪荡漾，梦境如画卷展开，我与自然融为一体，沉静在这片宁静的美丽之中，感受着生命的奇迹与温柔。"

rm hifigan_v2.onnx
rm -rf matcha-icefall-zh-baker

log "------------------------------------------------------------"
log "vits-piper-en_US-amy-low"
log "------------------------------------------------------------"
curl -O -SL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2
rm vits-piper-en_US-amy-low.tar.bz2

$EXE \
  --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
  --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
  --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
  --debug=1 \
  --output-filename=./tts/amy.wav \
  "“Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.” The sun shone bleakly in the sky, its meager light struggling to penetrate the thick foliage of the forest. Birds sang their songs up in the crowns of the trees, fluttering from one branch to the other. A blanket of total tranquility lied over the forest. The peace was only broken by the steady gallop of the horses of the soldiers who were traveling to their upcoming knighting the morrow at Camelot, and rowdy conversation. “Finally we will get what we deserve,” “It’s been about time,” Perceval agreed. “We’ve been risking our arses for the past two years. It’s the least they could give us.” Merlin remained ostensibly silent, refusing to join the verbal parade of self-aggrandizing his fellow soldiers have engaged in. He found it difficult to happy about anything, when even if they had won the war, he had lost everything else in the process."

file ./tts/amy.wav
rm -rf vits-piper-en_US-amy-low

log "------------------------------------------------------------"
log "vits-ljs test"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-ljs.tar.bz2
curl -SL -O $repo_url
tar xvf vits-ljs.tar.bz2
rm vits-ljs.tar.bz2
repo=vits-ljs

log "Start testing ${repo_url}"

$EXE \
  --vits-model=$repo/vits-ljs.onnx \
  --vits-lexicon=$repo/lexicon.txt \
  --vits-tokens=$repo/tokens.txt \
  --output-filename=./tts/vits-ljs.wav \
  'liliana, the most beautiful and lovely assistant of our team!'

ls -lh ./tts

rm -rfv $repo

log "------------------------------------------------------------"
log "vits-vctk test"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2
curl -SL -O $repo_url
tar xvf vits-vctk.tar.bz2
rm vits-vctk.tar.bz2
repo=vits-vctk

log "Start testing ${repo_url}"

for sid in 0 10 90; do
  $EXE \
    --vits-model=$repo/vits-vctk.onnx \
    --vits-lexicon=$repo/lexicon.txt \
    --vits-tokens=$repo/tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-vctk-${sid}.wav \
    'liliana, the most beautiful and lovely assistant of our team!'
done

rm -rfv $repo

ls -lh tts/

log "------------------------------------------------------------"
log "vits-zh-aishell3"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
curl -SL -O $repo_url
tar xvf vits-zh-aishell3.tar.bz2
rm vits-zh-aishell3.tar.bz2
repo=vits-zh-aishell3

log "Start testing ${repo_url}"

for sid in 0 10 90; do
  $EXE \
    --vits-model=$repo/vits-aishell3.onnx \
    --vits-lexicon=$repo/lexicon.txt \
    --vits-tokens=$repo/tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-aishell3-${sid}.wav \
    '林美丽最美丽'
done

rm -rfv $repo

ls -lh ./tts/
