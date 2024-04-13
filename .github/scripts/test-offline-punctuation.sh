#!/usr/bin/env bash

set -ex

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Download the punctuation model                             "
log "------------------------------------------------------------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
repo=sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
ls -lh $repo

$EXE \
 --debug=1 \
 --ct-transformer=$repo/model.onnx \
 "这是一个测试你好吗How are you我很好thank you are you ok谢谢你"

$EXE \
 --debug=1 \
 --ct-transformer=$repo/model.onnx \
 "我们都是木头人不会说话不会动"

$EXE \
 --debug=1 \
 --ct-transformer=$repo/model.onnx \
 "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry"

rm -rf $repo
