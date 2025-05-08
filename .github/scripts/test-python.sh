#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

log "test offline dolphin ctc"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
rm sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2

python3 ./python-api-examples/offline-dolphin-ctc-decode-files.py

rm -rf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

log "test offline speech enhancement (GTCRN)"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav
python3 ./python-api-examples/offline-speech-enhancement-gtcrn.py
ls -lh *.wav

log "test offline zipformer (byte-level bpe, Chinese+English)"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
tar xvf sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2
rm sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2

repo=sherpa-onnx-zipformer-zh-en-2023-11-22

./python-api-examples/offline-decode-files.py  \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-34-avg-19.int8.onnx \
  --decoder=$repo/decoder-epoch-34-avg-19.onnx \
  --joiner=$repo/joiner-epoch-34-avg-19.int8.onnx \
  --num-threads=2 \
  --decoding-method=greedy_search \
  --debug=true \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/2.wav

rm -rf sherpa-onnx-zipformer-zh-en-2023-11-22

log "test offline Moonshine"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

python3 ./python-api-examples/offline-moonshine-decode-files.py

rm -rf sherpa-onnx-moonshine-tiny-en-int8

log "test offline speaker diarization"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

python3 ./python-api-examples/offline-speaker-diarization.py

rm -rf *.wav *.onnx ./sherpa-onnx-pyannote-segmentation-3-0


log "test_clustering"
pushd /tmp/
mkdir test-cluster
cd test-cluster
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
git clone https://github.com/csukuangfj/sr-data
popd

python3 ./sherpa-onnx/python/tests/test_fast_clustering.py

rm -rf /tmp/test-cluster

export GIT_CLONE_PROTECTION_ACTIVE=false

log "test offline SenseVoice CTC"
url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
name=$(basename $url)
repo=$(basename -s .tar.bz2 $name)

curl -SL -O $url
tar xvf $name
rm $name
ls -lh $repo
python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
tar xf dict.tar.bz2
rm dict.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

python3 ./python-api-examples/offline-sense-voice-ctc-decode-files-with-hr.py

rm -rf dict replace.fst test-hr.wav lexicon.txt

if [[ $(uname) == Linux ]]; then
  # It needs ffmpeg
  log  "generate subtitles (Chinese)"
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  python3 ./python-api-examples/generate-subtitles.py \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=$repo/model.onnx \
    --tokens=$repo/tokens.txt \
    --num-threads=2 \
    ./lei-jun-test.wav

  cat lei-jun-test.srt

  rm lei-jun-test.wav

  log  "generate subtitles (English)"
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav

  python3 ./python-api-examples/generate-subtitles.py \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=$repo/model.onnx \
    --tokens=$repo/tokens.txt \
    --num-threads=2 \
    ./Obama.wav

  cat Obama.srt
  rm Obama.wav
  rm silero_vad.onnx
fi
rm -rf $repo

log "test offline TeleSpeech CTC"
url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04.tar.bz2
name=$(basename $url)
repo=$(basename -s .tar.bz2 $name)

curl -SL -O $url
tar xvf $name
rm $name
ls -lh $repo
python3 ./python-api-examples/offline-telespeech-ctc-decode-files.py
rm -rf $repo

log "test online NeMo CTC"

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-80ms.tar.bz2
name=$(basename $url)
repo=$(basename -s .tar.bz2 $name)

curl -SL -O $url
tar xvf $name
rm $name
ls -lh $repo
python3 ./python-api-examples/online-nemo-ctc-decode-files.py
rm -rf $repo

log "test offline punctuation"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
repo=sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
ls -lh $repo

python3 ./python-api-examples/add-punctuation.py

rm -rf $repo

log "test online punctuation"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
repo=sherpa-onnx-online-punct-en-2024-08-06
ls -lh $repo

python3 ./python-api-examples/add-punctuation-online.py

rm -rf $repo

log "test audio tagging"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
 python3 ./python-api-examples/audio-tagging-from-a-file.py
rm -rf sherpa-onnx-zipformer-audio-tagging-2024-04-09


log "test streaming zipformer2 ctc HLG decoding"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
repo=sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

python3 ./python-api-examples/online-zipformer-ctc-hlg-decode-file.py \
  --debug 1 \
  --tokens ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt \
  --graph ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst \
  --model ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/0.wav

rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18


mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

pushd $dir

repo=$dir/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18
mkdir -p $repo
cd $repo
mkdir exp-ctc-rnnt-small
cd exp-ctc-rnnt-small
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/exp-ctc-rnnt-small/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx
cd ..
mkdir -p data/lang_bpe_500
cd data/lang_bpe_500
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/data/lang_bpe_500/tokens.txt
cd ../..
mkdir test_wavs
cd test_wavs

curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/0.wav
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/1.wav
curl -LS -O https://huggingface.co/csukuangfj/icefall-asr-librispeech-streaming-zipformer-small-2024-03-18/resolve/main/test_wavs/8k.wav
popd

python3 ./python-api-examples/online-decode-files.py \
  --tokens=$repo/data/lang_bpe_500/tokens.txt \
  --zipformer2-ctc=$repo/exp-ctc-rnnt-small/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

rm -rf $repo

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

wenet_models=(
# sherpa-onnx-zh-wenet-aishell
# sherpa-onnx-zh-wenet-aishell2
# sherpa-onnx-zh-wenet-wenetspeech
# sherpa-onnx-zh-wenet-multi-cn
sherpa-onnx-en-wenet-librispeech
# sherpa-onnx-en-wenet-gigaspeech
)

for name in ${wenet_models[@]}; do
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$name.tar.bz2
  curl -SL -O $repo_url
  tar xvf $name.tar.bz2
  rm $name.tar.bz2
  repo=$name
  log "Start testing ${repo_url}"

  if false; then
    # offline wenet ctc models are not supported by onnxruntime >= 1.18
    python3 ./python-api-examples/offline-decode-files.py \
      --tokens=$repo/tokens.txt \
      --wenet-ctc=$repo/model.onnx \
      $repo/test_wavs/0.wav \
      $repo/test_wavs/1.wav \
      $repo/test_wavs/8k.wav
  fi

  python3 ./python-api-examples/online-decode-files.py \
    --tokens=$repo/tokens.txt \
    --wenet-ctc=$repo/model-streaming.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

  python3 sherpa-onnx/python/tests/test_online_recognizer.py --verbose

  rm -rf $repo
done

log "Offline TTS test"
# test waves are saved in ./tts
mkdir ./tts

log "kokoro-multi-lang-v1_0 test"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xf kokoro-multi-lang-v1_0.tar.bz2
rm kokoro-multi-lang-v1_0.tar.bz2

python3 ./python-api-examples/offline-tts.py \
  --debug=1 \
  --kokoro-model=./kokoro-multi-lang-v1_0/model.onnx \
  --kokoro-voices=./kokoro-multi-lang-v1_0/voices.bin \
  --kokoro-tokens=./kokoro-multi-lang-v1_0/tokens.txt \
  --kokoro-data-dir=./kokoro-multi-lang-v1_0/espeak-ng-data \
  --kokoro-dict-dir=./kokoro-multi-lang-v1_0/dict \
  --kokoro-lexicon=./kokoro-multi-lang-v1_0/lexicon-us-en.txt,./kokoro-multi-lang-v1_0/lexicon-zh.txt \
  --num-threads=2 \
  --sid=18 \
  --output-filename="./tts/kokoro-18-zh-en.wav" \
  "中英文语音合成测试。This is generated by next generation Kaldi using Kokoro without Misaki. 你觉得中英文说的如何呢？"

rm -rf kokoro-multi-lang-v1_0

log "kokoro-en-v0_19 test"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

python3 ./python-api-examples/offline-tts.py \
  --debug=1 \
  --kokoro-model=./kokoro-en-v0_19/model.onnx \
  --kokoro-voices=./kokoro-en-v0_19/voices.bin \
  --kokoro-tokens=./kokoro-en-v0_19/tokens.txt \
  --kokoro-data-dir=./kokoro-en-v0_19/espeak-ng-data \
  --num-threads=2 \
  --sid=10 \
  --output-filename="./tts/kokoro-10.wav" \
  "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be  a statesman, a businessman, an official, or a scholar."

rm -rf kokoro-en-v0_19

log "matcha-ljspeech-en test"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

python3 ./python-api-examples/offline-tts.py \
  --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --matcha-vocoder=./vocos-22khz-univ.onnx \
  --matcha-tokens=./matcha-icefall-en_US-ljspeech/tokens.txt \
  --matcha-data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
  --output-filename=./tts/test-matcha-ljspeech-en.wav \
  --num-threads=2 \
 "Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar."

rm vocos-22khz-univ.onnx
rm -rf matcha-icefall-en_US-ljspeech

log "matcha-baker-zh test"

curl -O -SL https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

python3 ./python-api-examples/offline-tts.py \
 --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
 --matcha-vocoder=./vocos-22khz-univ.onnx \
 --matcha-lexicon=./matcha-icefall-zh-baker/lexicon.txt \
 --matcha-tokens=./matcha-icefall-zh-baker/tokens.txt \
 --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
 --matcha-dict-dir=./matcha-icefall-zh-baker/dict \
 --output-filename=./tts/test-matcha-baker-zh.wav \
 "某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。"

rm -rf matcha-icefall-zh-baker
rm vocos-22khz-univ.onnx

log "vits-ljs test"

curl -LS -O https://huggingface.co/csukuangfj/vits-ljs/resolve/main/vits-ljs.onnx
curl -LS -O https://huggingface.co/csukuangfj/vits-ljs/resolve/main/lexicon.txt
curl -LS -O https://huggingface.co/csukuangfj/vits-ljs/resolve/main/tokens.txt

python3 ./python-api-examples/offline-tts.py \
  --vits-model=./vits-ljs.onnx \
  --vits-lexicon=./lexicon.txt \
  --vits-tokens=./tokens.txt \
  --output-filename=./tts/vits-ljs.wav \
  'liliana, the most beautiful and lovely assistant of our team!'

ls -lh ./tts

rm -v vits-ljs.onnx ./lexicon.txt ./tokens.txt

log "vits-vctk test"
curl -LS -O https://huggingface.co/csukuangfj/vits-vctk/resolve/main/vits-vctk.onnx
curl -LS -O https://huggingface.co/csukuangfj/vits-vctk/resolve/main/lexicon.txt
curl -LS -O https://huggingface.co/csukuangfj/vits-vctk/resolve/main/tokens.txt

for sid in 0 10 90; do
  python3 ./python-api-examples/offline-tts.py \
    --vits-model=./vits-vctk.onnx \
    --vits-lexicon=./lexicon.txt \
    --vits-tokens=./tokens.txt \
    --sid=$sid \
    --output-filename=./tts/vits-vctk-${sid}.wav \
    'liliana, the most beautiful and lovely assistant of our team!'
done

rm -v vits-vctk.onnx ./lexicon.txt ./tokens.txt

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"

  log "vits-zh-aishell3"

  curl -LS -O https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/vits-aishell3.onnx
  curl -LS -O https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/lexicon.txt
  curl -LS -O https://huggingface.co/csukuangfj/vits-zh-aishell3/resolve/main/tokens.txt

  for sid in 0 10 90; do
    python3 ./python-api-examples/offline-tts.py \
      --vits-model=./vits-aishell3.onnx \
      --vits-lexicon=./lexicon.txt \
      --vits-tokens=./tokens.txt \
      --sid=$sid \
      --output-filename=./tts/vits-aishell3-${sid}.wav \
      '林美丽最美丽'
  done

  rm -v vits-aishell3.onnx ./lexicon.txt ./tokens.txt
fi

mkdir -p /tmp/icefall-models
dir=/tmp/icefall-models

log "Test streaming transducer models"

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"
  pushd $dir
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  curl -SL -O $repo_url
  tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
  repo=sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

  log "Start testing ${repo_url}"
  repo=$dir/$repo

  python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
  sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

  echo "sherpa_onnx version: $sherpa_onnx_version"

  pwd
  ls -lh

  ls -lh $repo
  popd

  python3 ./python-api-examples/online-decode-files.py \
    --tokens=$repo/tokens.txt \
    --encoder=$repo/encoder-epoch-99-avg-1.onnx \
    --decoder=$repo/decoder-epoch-99-avg-1.onnx \
    --joiner=$repo/joiner-epoch-99-avg-1.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/8k.wav

  python3 ./python-api-examples/online-decode-files.py \
    --tokens=$repo/tokens.txt \
    --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
    --decoder=$repo/decoder-epoch-99-avg-1.onnx \
    --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/3.wav \
    $repo/test_wavs/8k.wav

  ln -s $repo $PWD/

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

  python3 ./python-api-examples/inverse-text-normalization-online-asr.py

  python3 sherpa-onnx/python/tests/test_online_recognizer.py --verbose

  rm -rfv sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

  rm -rf $repo
fi

log "Test non-streaming transducer models"

pushd $dir
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
log "Download pretrained model and test-data from $repo_url"

curl -SL -O $repo_url
tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
repo=$dir/sherpa-onnx-zipformer-en-2023-04-01

popd

ls -lh $repo

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.int8.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

lm_repo_url=https://huggingface.co/ezerhouni/icefall-librispeech-rnn-lm
log "Download pre-trained RNN-LM model from ${lm_repo_url}"
GIT_LFS_SKIP_SMUDGE=1 git clone $lm_repo_url
lm_repo=$(basename $lm_repo_url)
pushd $lm_repo
git lfs pull --include "exp/no-state-epoch-99-avg-1.onnx"
popd

bigram_repo_url=https://huggingface.co/vsd-vector/librispeech_bigram_sherpa-onnx-zipformer-large-en-2023-06-26
log "Download bi-gram LM from ${bigram_repo_url}"
GIT_LFS_SKIP_SMUDGE=1 git clone $bigram_repo_url
bigramlm_repo=$(basename $bigram_repo_url)
pushd $bigramlm_repo
git lfs pull --include "2gram.fst"
popd

log "Perform offline decoding with RNN-LM and LODR"
python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder-epoch-99-avg-1.onnx \
  --decoder=$repo/decoder-epoch-99-avg-1.onnx \
  --joiner=$repo/joiner-epoch-99-avg-1.onnx \
  --lm=$lm_repo/exp/no-state-epoch-99-avg-1.onnx \
  --lodr-fst=$bigramlm_repo/2gram.fst \
  --lodr-scale=-0.5 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo $lm_repo $bigramlm_repo

log "Test non-streaming paraformer models"

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"
  pushd $dir
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  curl -SL -O $repo_url
  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  log "Start testing ${repo_url}"
  repo=$dir/sherpa-onnx-paraformer-zh-2023-09-14

  ls -lh $repo
  popd

  python3 ./python-api-examples/offline-decode-files.py \
    --tokens=$repo/tokens.txt \
    --paraformer=$repo/model.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav \
    $repo/test_wavs/8k.wav

  python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

  ln -s $repo $PWD/

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

  python3 ./python-api-examples/inverse-text-normalization-offline-asr.py

  rm -rfv sherpa-onnx-paraformer-zh-2023-09-14

  rm -rf $repo
fi

log "Test non-streaming NeMo CTC models"

pushd $dir
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
rm sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2

log "Start testing ${repo_url}"
repo=$dir/sherpa-onnx-nemo-ctc-en-citrinet-512

ls -lh $repo
popd

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --nemo-ctc=$repo/model.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 ./python-api-examples/offline-decode-files.py \
  --tokens=$repo/tokens.txt \
  --nemo-ctc=$repo/model.int8.onnx \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

python3 sherpa-onnx/python/tests/test_offline_recognizer.py --verbose

rm -rf $repo

# test text2token
git clone https://github.com/pkufool/sherpa-test-data /tmp/sherpa-test-data

python3 sherpa-onnx/python/tests/test_text2token.py --verbose

rm -rf /tmp/sherpa-test-data

dir=/tmp/onnx-models
mkdir -p $dir

log "Test keyword spotting models"

python3 -c "import sherpa_onnx; print(sherpa_onnx.__file__)"
sherpa_onnx_version=$(python3 -c "import sherpa_onnx; print(sherpa_onnx.__version__)")

echo "sherpa_onnx version: $sherpa_onnx_version"

pwd
ls -lh

if [[ x$OS != x'windows-latest' ]]; then
  echo "OS: $OS"

  repo=sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
  log "Start testing ${repo}"

  curl -LS -O https://github.com/pkufool/keyword-spotting-models/releases/download/v0.1/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz
  tar xf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz
  rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz

  ls -lh $repo

  python3 ./python-api-examples/keyword-spotter.py

  python3 sherpa-onnx/python/tests/test_keyword_spotter.py --verbose

  rm -rf $repo
fi

rm -r $dir
