#!/usr/bin/env bash

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

export GIT_CLONE_PROTECTION_ACTIVE=false

echo "EXE is $EXE"
echo "PATH: $PATH"

which $EXE

log "------------------------------------------------------------"
log "Run NeMo transducer (English)"
log "------------------------------------------------------------"
repo_url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms.tar.bz2
download_and_extract $repo_url

log "Start testing ${repo_url}"

repo=sherpa-onnx-nemo-streaming-fast-conformer-transducer-en-80ms

# decode-file-c-api hard-codes these model paths and ignores any command-line
# arguments (its main() takes none). The generic sherpa-onnx CLI needs them.
# Passing the paths works for both: the CLI uses them, the C example ignores
# them.
time $EXE \
  --tokens=$repo/tokens.txt \
  --encoder=$repo/encoder.onnx \
  --decoder=$repo/decoder.onnx \
  --joiner=$repo/joiner.onnx \
  $repo/test_wavs/0.wav

rm -rf $repo
