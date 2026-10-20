#!/usr/bin/env bash
#
# Tests every python-api-examples/test-offline-tts-*.py example.
#
# These cover the two text-frontend paths that replaced the removed espeak-ng
# support:
#
#   * *-lexicon.py     -- pass a lexicon file, no external phonemizer needed
#   * *-phonemize.py   -- phonemize outside sherpa-onnx (piper_phonemize,
#                         pypinyin, misaki) and pass phoneme_codepoints
#   * *-tokens.py      -- same as *-phonemize.py but pass token strings via
#                         GenerationConfig.tokens (sherpa-onnx adds the
#                         bos/pad/eos markers itself)
#
# Requires: piper_phonemize, pypinyin, misaki[zh], kaldifst, soundfile
#
# Note: misaki keeps its Chinese dependencies behind the "zh" extra, so
# "misaki[zh]" (not bare "misaki") is required for `from misaki import zh`.
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

python3 -m pip install --upgrade soundfile
python3 -m pip install --upgrade pypinyin 'misaki[zh]' kaldifst
python3 -m pip install --upgrade piper_phonemize -f https://k2-fsa.github.io/icefall/piper_phonemize.html

# Import exactly what the examples import -- bare "import misaki" does NOT
# exercise misaki.zh, which is where the extras-gated deps (cn2an, jieba,
# pypinyin-dict) are needed.
python3 -c "
import kaldifst
import numpy
import soundfile
import piper_phonemize
from pypinyin import Style, lazy_pinyin
from pypinyin.contrib.tone_convert import to_finals_tone3, to_initials
from misaki import zh
print('phonemizer deps OK')
"

# test waves are saved in ./tts
mkdir -p ./tts

LEXICON=https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/lexicon-en-us.txt
VOCOS_22=https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx
VOCOS_16=https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-16khz-univ.onnx
VOCOS_24=https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

run() {
  # run <example.py> <wav produced by the example>
  local ex=$1
  local wav=$2
  log "Run $ex"
  python3 ./python-api-examples/$ex
  mv -v "$wav" ./tts/
}

log "------------------------------------------------------------"
log "coqui (character frontend)"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
run test-offline-tts-coqui-phonemize.py test-coqui-phonemize.wav
rm -rf vits-coqui-de-css10

log "------------------------------------------------------------"
log "inflect (vits-ext)"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-inflect-en-nano-v2.tar.bz2
download $LEXICON
run test-offline-tts-inflect-lexicon.py test-inflect-lexicon.wav
run test-offline-tts-inflect-phonemize.py test-inflect-phonemize.wav
rm -rf vits-inflect-en-nano-v2 lexicon-en-us.txt

log "------------------------------------------------------------"
log "kitten"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-micro-en-v0_8.tar.bz2
download $LEXICON
run test-offline-tts-kitten-lexicon.py test-kitten-lexicon.wav
run test-offline-tts-kitten-phonemize.py test-kitten-phonemize.wav
rm -rf kitten-micro-en-v0_8 lexicon-en-us.txt

log "------------------------------------------------------------"
log "kokoro en"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
download $LEXICON
run test-offline-tts-kokoro-en-lexicon.py test-kokoro-en-lexicon.wav
run test-offline-tts-kokoro-en-phonemize.py test-kokoro-en-phonemize.wav
rm -rf kokoro-en-v0_19 lexicon-en-us.txt

log "------------------------------------------------------------"
log "kokoro multi-lang"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
run test-offline-tts-kokoro-multilang-lexicon.py test-kokoro-multilang-lexicon.wav
run test-offline-tts-kokoro-multilang-phonemize.py test-kokoro-multilang-phonemize.wav
rm -rf kokoro-multi-lang-v1_0

log "------------------------------------------------------------"
log "matcha en"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
download $VOCOS_22
download $LEXICON
run test-offline-tts-matcha-en-lexicon.py test-matcha-en-lexicon.wav
run test-offline-tts-matcha-en-phonemize.py test-matcha-en-phonemize.wav
rm -rf matcha-icefall-en_US-ljspeech lexicon-en-us.txt

log "------------------------------------------------------------"
log "matcha zh"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
run test-offline-tts-matcha-zh-lexicon.py test-matcha-zh-lexicon.wav
run test-offline-tts-matcha-zh-phonemize.py test-matcha-zh-phonemize.wav
rm -rf matcha-icefall-zh-baker

log "------------------------------------------------------------"
log "matcha zh-en"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-en.tar.bz2
download $VOCOS_16
download $LEXICON
run test-offline-tts-matcha-zh-en-lexicon.py test-matcha-zh-en-lexicon.wav
run test-offline-tts-matcha-zh-en-phonemize.py test-matcha-zh-en-phonemize.wav
rm -rf matcha-icefall-zh-en vocos-16khz-univ.onnx lexicon-en-us.txt

log "------------------------------------------------------------"
log "piper (vits)"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
download $LEXICON
run test-offline-tts-piper-lexicon.py test-piper-lexicon.wav
run test-offline-tts-piper-phonemize.py test-piper-phonemize.wav
run test-offline-tts-piper-tokens.py test-piper-tokens.wav
rm -rf vits-piper-en_US-amy-low lexicon-en-us.txt

log "------------------------------------------------------------"
log "zipvoice"
log "------------------------------------------------------------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2
download $VOCOS_24
download $LEXICON
run test-offline-tts-zipvoice-lexicon.py test-zipvoice-lexicon.wav
run test-offline-tts-zipvoice-phonemize.py test-zipvoice-phonemize.wav
rm -rf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia vocos_24khz.onnx lexicon-en-us.txt

# the 22 kHz vocoder is shared by matcha-en and matcha-zh
rm -f vocos-22khz-univ.onnx

log "All test-offline-tts examples passed"
ls -lh ./tts/
