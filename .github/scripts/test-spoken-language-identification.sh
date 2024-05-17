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

names=(
tiny
base
small
medium
)

# all_language_codes=bo,ml,tt,fa,sl,bg,sn,sr,tl,km,ln,mr,hr,eu,ro,ba,bs,pl,as,nn,sk,ko,oc,ar,uz,pa,tg,mk,kk,hi,ha,uk,is,de,el,ja,yo,be,so,tk,id,sa,ru,yi,en,am,cs,ne,la,sv,su,pt,mi,ca,sd,hy,haw,fi,et,kn,da,lt,it,nl,he,mg,ur,tr,af,br,bn,ta,no,my,si,mt,th,gl,sw,mn,jw,ms,ps,fo,ka,hu,zh,ht,az,fr,lo,sq,gu,cy,lv,es,lb,te,vi

log "Download test waves"
waves=(
ar-arabic.wav
bg-bulgarian.wav
cs-czech.wav
da-danish.wav
# de-german.wav
# el-greek.wav
# en-english.wav
# es-spanish.wav
# fa-persian.wav
# fi-finnish.wav
# fr-french.wav
# hi-hindi.wav
# hr-croatian.wav
# id-indonesian.wav
# it-italian.wav
# ja-japanese.wav
# ko-korean.wav
# nl-dutch.wav
# no-norwegian.wav
# po-polish.wav
# pt-portuguese.wav
# ro-romanian.wav
# ru-russian.wav
# sk-slovak.wav
# sv-swedish.wav
# ta-tamil.wav
# tl-tagalog.wav
# tr-turkish.wav
# uk-ukrainian.wav
# zh-chinese.wav
)

for wav in ${waves[@]}; do
  echo "Downloading $wav"
  curl -SL -O https://hf-mirror.com/spaces/k2-fsa/spoken-language-identification/resolve/main/test_wavs/$wav
  ls -lh *.wav
done

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
tar xvf spoken-language-identification-test-wavs.tar.bz2
rm spoken-language-identification-test-wavs.tar.bz2
data=spoken-language-identification-test-wavs

for name in ${names[@]}; do
  log "------------------------------------------------------------"
  log "Run $name"
  log "------------------------------------------------------------"

  repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-whisper-$name
  log "Start testing ${repo_url}"
  repo=$(basename $repo_url)
  log "Download pretrained model and test-data from $repo_url"

  GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
  pushd $repo
  git lfs pull --include "*.onnx"
  # git lfs pull --include "*.ort"
  ls -lh *.onnx
  popd

  for wav in ${waves[@]}; do
    log "test fp32 onnx"

    time $EXE \
      --whisper-encoder=$repo/${name}-encoder.onnx \
      --whisper-decoder=$repo/${name}-decoder.onnx \
      $data/$wav

    log "test int8 onnx"

    time $EXE \
      --whisper-encoder=$repo/${name}-encoder.int8.onnx \
      --whisper-decoder=$repo/${name}-decoder.int8.onnx \
      $data/$wav
  done
  rm -rf $repo
done
