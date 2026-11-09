#!/usr/bin/env bash
#
# Test group: offline (non-streaming) ASR model zoo.
#
# One of several groups run in parallel by .github/workflows/run-python-test.yaml.
# Run .github/scripts/test-python.sh to run every group sequentially.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

log "test Cohere Transcribe"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2
python3 ./python-api-examples/offline-cohere-transcribe-decode-files.py
rm -rf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01

log "test Qwen3 ASR"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

python3 ./python-api-examples/offline-qwen3-asr-decode-files.py \
  --conv-frontend=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx \
  --encoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx \
  --decoder=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx \
  --tokenizer=./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer \
  --max-new-tokens=128 \
  --num-threads=2 \
  ./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav

rm -rf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25

log "test Moonshine v2"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2

ls -lh sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27

python3 ./python-api-examples/offline-moonshine-decode-files-v2.py

rm -rf  sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27

log "test offline Moonshine"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

python3 ./python-api-examples/offline-moonshine-decode-files.py

rm -rf sherpa-onnx-moonshine-tiny-en-int8

log "test FireRedASR CTC"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

python3 ./python-api-examples/offline-fire-red-asr-ctc-decode-files.py

rm -rf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25

log "test FireRedASR AED"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2

python3 ./python-api-examples/offline-fire-red-asr-decode-files.py

rm -rf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16

log "test Google MedASR"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
ls -lh sherpa-onnx-medasr-ctc-en-int8-2025-12-25

ls -lh sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs

python3 ./python-api-examples/offline-medasr-ctc-decode-files.py
rm -rf sherpa-onnx-medasr-ctc-en-int8-2025-12-25

log "test omnilingual ASR"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
ls -lh sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12

python3 ./python-api-examples/offline-omnilingual-asr-ctc-decode-files.py

rm -rf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12

log "test nemo canary"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
python3 ./python-api-examples/offline-nemo-canary-decode-files.py
rm -rf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8

log "test offline dolphin ctc"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2

python3 ./python-api-examples/offline-dolphin-ctc-decode-files.py

rm -rf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

log "test offline zipformer (byte-level bpe, Chinese+English)"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-zh-en-2023-11-22.tar.bz2

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

log "test offline SenseVoice CTC"
url=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2
name=$(basename $url)
repo=$(basename -s .tar.bz2 $name)

download_and_extract $url
ls -lh $repo
python3 ./python-api-examples/offline-sense-voice-ctc-decode-files.py

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

python3 ./python-api-examples/offline-sense-voice-ctc-decode-files-with-hr.py

rm -rf dict replace.fst test-hr.wav lexicon.txt

if command -v ffmpeg &> /dev/null; then
  # It needs ffmpeg
  log  "generate subtitles (Chinese)"
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav

  python3 ./python-api-examples/generate-subtitles.py \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=$repo/model.int8.onnx \
    --tokens=$repo/tokens.txt \
    --num-threads=2 \
    ./lei-jun-test.wav

  cat lei-jun-test.srt

  rm lei-jun-test.wav

  log  "generate subtitles (English)"
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav

  python3 ./python-api-examples/generate-subtitles.py \
    --silero-vad-model=./silero_vad.onnx \
    --sense-voice=$repo/model.int8.onnx \
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

download_and_extract $url
ls -lh $repo
python3 ./python-api-examples/offline-telespeech-ctc-decode-files.py
rm -rf $repo

log "test-python-asr-offline.sh passed"
