#!/usr/bin/env bash
#
# Test group: forced alignment (ForcedAligner Python API).
#
# One of several groups run in parallel by .github/workflows/run-python-test.yaml.
# Run .github/scripts/test-python.sh to run every group sequentially.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export GIT_CLONE_PROTECTION_ACTIVE=false

log "test forced alignment (English)"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
python3 ./python-api-examples/forced-alignment-english.py
grep -q 'name = "phones"' en.TextGrid
grep -q 'name = "words"' en.TextGrid

log "test forced alignment (Chinese)"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
python3 ./python-api-examples/forced-alignment-chinese.py
grep -q 'name = "phones"' zh.TextGrid
grep -q 'name = "words"' zh.TextGrid

log "test textgrid to word wavs"
python3 ./python-api-examples/textgrid-to-word-wavs.py \
  --wav english-mfa-v3.1.0/en.wav \
  --textgrid en.TextGrid \
  --out-dir word-clips-en
python3 ./python-api-examples/textgrid-to-word-wavs.py \
  --wav zh-aishell3-pinyin/zh.wav \
  --textgrid zh.TextGrid \
  --out-dir word-clips-zh
test -f word-clips-en/w01.wav
test -f word-clips-en/manifest.json
test -f word-clips-zh/w01.wav
test -f word-clips-zh/manifest.json

rm -rf english-mfa-v3.1.0 zh-aishell3-pinyin
