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

log "-----------------------------------------------------------------"
log "Run Nemo fast conformer hybrid transducer ctc models (CTC branch)"
log "-----------------------------------------------------------------"

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
name=$(basename $url)
curl -SL -O $url
tar xvf $name
rm $name
repo=$(basename -s .tar.bz2 $name)
ls -lh $repo

log "test $repo"
test_wavs=(
de-german.wav
es-spanish.wav
hr-croatian.wav
po-polish.wav
uk-ukrainian.wav
en-english.wav
fr-french.wav
it-italian.wav
ru-russian.wav
)
for w in ${test_wavs[@]}; do
  time $EXE \
    --tokens=$repo/tokens.txt \
    --nemo-ctc-model=$repo/model.onnx \
    --debug=1 \
    $repo/test_wavs/$w
done

rm -rf $repo

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-en-24500.tar.bz2
name=$(basename $url)
curl -SL -O $url
tar xvf $name
rm $name
repo=$(basename -s .tar.bz2 $name)
ls -lh $repo

log "Test $repo"

time $EXE \
  --tokens=$repo/tokens.txt \
  --nemo-ctc-model=$repo/model.onnx \
  --debug=1 \
  $repo/test_wavs/en-english.wav

rm -rf $repo

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-es-1424.tar.bz2
name=$(basename $url)
curl -SL -O $url
tar xvf $name
rm $name
repo=$(basename -s .tar.bz2 $name)
ls -lh $repo

log "test $repo"

time $EXE \
  --tokens=$repo/tokens.txt \
  --nemo-ctc-model=$repo/model.onnx \
  --debug=1 \
  $repo/test_wavs/es-spanish.wav

rm -rf $repo

url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288.tar.bz2
name=$(basename $url)
curl -SL -O $url
tar xvf $name
rm $name
repo=$(basename -s .tar.bz2 $name)
ls -lh $repo

log "Test $repo"

test_wavs=(
en-english.wav
de-german.wav
fr-french.wav
es-spanish.wav
)

for w in ${test_wavs[@]}; do
  time $EXE \
    --tokens=$repo/tokens.txt \
    --nemo-ctc-model=$repo/model.onnx \
    --debug=1 \
    $repo/test_wavs/$w
done

rm -rf $repo

log "------------------------------------------------------------"
log "Run Wenet models"
log "------------------------------------------------------------"
wenet_models=(
sherpa-onnx-zh-wenet-aishell
# sherpa-onnx-zh-wenet-aishell2
# sherpa-onnx-zh-wenet-wenetspeech
# sherpa-onnx-zh-wenet-multi-cn
sherpa-onnx-en-wenet-librispeech
# sherpa-onnx-en-wenet-gigaspeech
)
for name in ${wenet_models[@]}; do
  repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$name.tar.bz2
  log "Start testing ${repo_url}"
  repo=$name
  log "Download pretrained model and test-data from $repo_url"
  curl -SL -O $repo_url
  tar xvf $name.tar.bz2
  rm $name.tar.bz2

  log "test float32 models"
  time $EXE \
    --tokens=$repo/tokens.txt \
    --wenet-ctc-model=$repo/model.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  log "test int8 models"
  time $EXE \
    --tokens=$repo/tokens.txt \
    --wenet-ctc-model=$repo/model.int8.onnx \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/8k.wav

  rm -rf $repo
done

log "------------------------------------------------------------"
log "Run tdnn yesno (Hebrew)"
log "------------------------------------------------------------"
url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-tdnn-yesno.tar.bz2
curl -SL -O $url
tar xvf sherpa-onnx-tdnn-yesno.tar.bz2
rm sherpa-onnx-tdnn-yesno.tar.bz2
log "Start testing ${url}"
repo=sherpa-onnx-tdnn-yesno
log "Download pretrained model and test-data from $url"

log "test float32 models"
time $EXE \
  --sample-rate=8000 \
  --feat-dim=23 \
  \
  --tokens=$repo/tokens.txt \
  --tdnn-model=$repo/model-epoch-14-avg-2.onnx \
  $repo/test_wavs/0_0_0_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_0_0_0_1_0.wav \
  $repo/test_wavs/0_0_1_0_0_1_1_1.wav \
  $repo/test_wavs/0_0_1_0_1_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_1_1_0.wav

log "test int8 models"
time $EXE \
  --sample-rate=8000 \
  --feat-dim=23 \
  \
  --tokens=$repo/tokens.txt \
  --tdnn-model=$repo/model-epoch-14-avg-2.int8.onnx \
  $repo/test_wavs/0_0_0_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_0_0_0_1_0.wav \
  $repo/test_wavs/0_0_1_0_0_1_1_1.wav \
  $repo/test_wavs/0_0_1_0_1_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_0_0_1.wav \
  $repo/test_wavs/0_0_1_1_0_1_1_0.wav

rm -rf $repo

log "------------------------------------------------------------"
log "Run Citrinet (stt_en_citrinet_512, English)"
log "------------------------------------------------------------"

repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
curl -SL -O $repo_url
tar xvf sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
rm sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
log "Start testing ${repo_url}"
repo=sherpa-onnx-nemo-ctc-en-citrinet-512
log "Download pretrained model and test-data from $repo_url"

time $EXE \
  --tokens=$repo/tokens.txt \
  --nemo-ctc-model=$repo/model.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

time $EXE \
  --tokens=$repo/tokens.txt \
  --nemo-ctc-model=$repo/model.int8.onnx \
  --num-threads=2 \
  $repo/test_wavs/0.wav \
  $repo/test_wavs/1.wav \
  $repo/test_wavs/8k.wav

rm -rf $repo

log "------------------------------------------------------------"
log "Run Librispeech zipformer CTC H/HL/HLG decoding (English)   "
log "------------------------------------------------------------"
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
curl -SL -O $repo_url
log "Start testing ${repo_url}"
tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
repo=sherpa-onnx-zipformer-ctc-en-2023-10-02
log "Download pretrained model and test-data from $repo_url"

graphs=(
$repo/H.fst
$repo/HL.fst
$repo/HLG.fst
)

for graph in ${graphs[@]}; do
  log "test float32 models with $graph"
  time $EXE \
    --model-type=zipformer2_ctc \
    --ctc.graph=$graph \
    --zipformer-ctc-model=$repo/model.onnx \
    --tokens=$repo/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav

  log "test int8 models with $graph"
  time $EXE \
    --model-type=zipformer2_ctc \
    --ctc.graph=$graph \
    --zipformer-ctc-model=$repo/model.int8.onnx \
    --tokens=$repo/tokens.txt \
    $repo/test_wavs/0.wav \
    $repo/test_wavs/1.wav \
    $repo/test_wavs/2.wav
done

rm -rf $repo
