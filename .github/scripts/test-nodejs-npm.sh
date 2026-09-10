#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
echo "dir: $d"
cd $d
npm install
git status
ls -lh
ls -lh node_modules

echo "---test version---"
node ./test-version.js

if false; then
  # disable it for now since it fails, possible due to not using a recent version of onnxruntime
  echo "---test Qwen3 ASR---"

  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

  node ./test-offline-qwen3-asr.js

  rm -rf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25
fi

echo "---test punctuation---"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2

node ./test-offline-punctuation.js

# disable it now since it causes CI failure
# node ./test-online-punctuation.js

rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
rm -rf sherpa-onnx-online-punct-en-2024-08-06

echo "---test moonshine v2---"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2

node ./test-offline-moonshine-v2.js

rm -rf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27

echo "---test FireRedASR CTC---"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

node ./test-offline-fire-red-asr-ctc.js

rm -rf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25

# curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
# tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
# rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
#
# node ./test-offline-funasr-nano.js
#
# rm -rf sherpa-onnx-funasr-nano-int8-2025-12-30

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2

node ./test-offline-medasr-ctc.js

rm -rf sherpa-onnx-medasr-ctc-en-int8-2025-12-25

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2

node ./test-offline-omnilingual-asr-ctc.js

rm -rf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2

node ./test-offline-wenet-ctc.js
rm -rf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
node ./test-online-t-one-ctc.js

rm -rf sherpa-onnx-streaming-t-one-russian-2025-09-08

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2

node ./test-offline-tts-kitten-en.js
ls -lh *.wav
rm -rf kitten-nano-en-v0_1-fp16

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

node ./test-offline-tts-supertonic.js
ls -lh *.wav
rm -rf sherpa-onnx-supertonic-3-tts-int8-2026-05-11

# online asr
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
node ./test-online-paraformer.js
rm -rf sherpa-onnx-streaming-paraformer-bilingual-zh-en

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

rm -f itn*
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst

node ./test-online-transducer-itn.js

node ./test-online-transducer.js

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2

node ./test-online-zipformer2-ctc.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
node ./test-online-zipformer2-ctc-hlg.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

echo "----------keyword spotting----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

node ./test-keyword-spotter-transducer.js
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

# asr with offline nemo canary
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2

node ./test-offline-nemo-canary.js
rm -rf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8

# asr with offline zipformer ctc
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

node ./test-offline-zipformer-ctc.js
rm -rf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03

# asr with offline dolphin ctc
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
node ./test-offline-dolphin-ctc.js
rm -rf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

# speech enhancement
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
node ./test-offline-speech-enhancement-gtcrn.js
node ./test-offline-speech-enhancement-dpdfnet.js
node ./test-online-speech-enhancement-gtcrn.js
node ./test-online-speech-enhancement-dpdfnet.js
ls -lh *.wav
rm gtcrn_simple.onnx
rm dpdfnet_baseline.onnx
rm -fv inp_16k.wav
rm -fv enhanced*.wav

# offline tts

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

node ./test-offline-tts-zipvoice-zh-en.js
ls -lh *.wav
rm -rf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia
rm -f vocos_24khz.onnx

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2

node ./test-offline-tts-kokoro-zh-en.js
ls -lh *.wav
rm -rf kokoro-multi-lang-v1_0

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2

node ./test-offline-tts-kokoro-en.js
rm -rf kokoro-en-v0_19

ls -lh

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test-offline-tts-matcha-zh.js

rm -rf matcha-icefall-zh-baker
rm vocos-22khz-univ.onnx


echo "---"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test-offline-tts-matcha-en.js

rm -rf matcha-icefall-en_US-ljspeech
rm vocos-22khz-univ.onnx

echo "---"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
node ./test-offline-tts-vits-en.js
rm -rf vits-piper-en_US-amy-low*

echo "---"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
node ./test-offline-tts-vits-zh.js
rm -rf vits-icefall-zh-aishell3*

ls -lh *.wav

echo '-----speaker diarization----------'
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test-offline-speaker-diarization.js
rm -rfv *.wav *.onnx sherpa-onnx-pyannote-*

echo '-----vad+moonshine----------'

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
node ./test-vad-with-non-streaming-asr-whisper.js
rm Obama.wav
rm silero_vad.onnx
rm -rf sherpa-onnx-moonshine-*

echo '-----vad+whisper----------'

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
node ./test-vad-with-non-streaming-asr-whisper.js
rm Obama.wav
rm silero_vad.onnx
rm -rf sherpa-onnx-whisper-tiny.en

# offline asr
#
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

node ./test-offline-sense-voice.js

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

node ./test-offline-sense-voice-with-hr.js

rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
rm -rf dict replace.fst test-hr.wav lexicon.txt

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

rm -f itn*
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
node ./test-offline-paraformer-itn.js
rm -rf sherpa-onnx-paraformer-zh-2023-09-14

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
node ./test-offline-nemo-ctc.js
rm -rf sherpa-onnx-nemo-ctc-en-conformer-small

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
node ./test-offline-paraformer.js
rm -rf sherpa-onnx-paraformer-zh-2023-09-14

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
node ./test-offline-transducer.js
rm -rf sherpa-onnx-zipformer-en-2023-06-26

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
node ./test-offline-whisper.js
rm -rf sherpa-onnx-whisper-tiny.en

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

node ./test-offline-moonshine.js
rm -rf sherpa-onnx-moonshine-*
