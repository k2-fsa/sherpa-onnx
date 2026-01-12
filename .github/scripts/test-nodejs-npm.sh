#!/usr/bin/env bash

set -ex

echo "dir: $d"
cd $d
npm install
git status
ls -lh
ls -lh node_modules

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
tar xvf sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2
rm sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

node ./test-offline-funasr-nano.js

rm -rf sherpa-onnx-funasr-nano-int8-2025-12-30

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2

node ./test-offline-medasr-ctc.js

rm -rf sherpa-onnx-medasr-ctc-en-int8-2025-12-25

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
tar xvf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
rm sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2

node ./test-offline-omnilingual-asr-ctc.js

rm -rf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
tar xvf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
rm sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2

node ./test-offline-wenet-ctc.js
rm -rf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
node ./test-online-t-one-ctc.js

rm -rf sherpa-onnx-streaming-t-one-russian-2025-09-08

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
tar xf kitten-nano-en-v0_1-fp16.tar.bz2
rm kitten-nano-en-v0_1-fp16.tar.bz2

node ./test-offline-tts-kitten-en.js
ls -lh *.wav
rm -rf kitten-nano-en-v0_1-fp16

# online asr
curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
node ./test-online-paraformer.js
rm -rf sherpa-onnx-streaming-paraformer-bilingual-zh-en

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

rm -f itn*
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst

node ./test-online-transducer-itn.js

node ./test-online-transducer.js

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2

node ./test-online-zipformer2-ctc.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
node ./test-online-zipformer2-ctc-hlg.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

echo "----------keyword spotting----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

node ./test-keyword-spotter-transducer.js
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

# asr with offline nemo canary
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
tar xvf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
rm sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2

node ./test-offline-nemo-canary.js
rm -rf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8

# asr with offline zipformer ctc
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

node ./test-offline-zipformer-ctc.js
rm -rf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03

# asr with offline dolphin ctc
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
rm sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
node ./test-offline-dolphin-ctc.js
rm -rf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

# speech enhancement
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
node ./test-offline-speech-enhancement-gtcrn.js
ls -lh *.wav
rm gtcrn_simple.onnx
rm inp_16k.wav
rm enhanced-16k.wav

# offline tts

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2
tar xf kokoro-multi-lang-v1_0.tar.bz2
rm kokoro-multi-lang-v1_0.tar.bz2

node ./test-offline-tts-kokoro-zh-en.js
ls -lh *.wav
rm -rf kokoro-multi-lang-v1_0

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

node ./test-offline-tts-kokoro-en.js
rm -rf kokoro-en-v0_19

ls -lh

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test-offline-tts-matcha-zh.js

rm -rf matcha-icefall-zh-baker
rm vocos-22khz-univ.onnx


echo "---"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test-offline-tts-matcha-en.js

rm -rf matcha-icefall-en_US-ljspeech
rm vocos-22khz-univ.onnx

echo "---"

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2
node ./test-offline-tts-vits-en.js
rm -rf vits-piper-en_US-amy-low*

echo "---"

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
node ./test-offline-tts-vits-zh.js
rm -rf vits-icefall-zh-aishell3*

ls -lh *.wav

echo '-----speaker diarization----------'
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test-offline-speaker-diarization.js
rm -rfv *.wav *.onnx sherpa-onnx-pyannote-*

echo '-----vad+moonshine----------'

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
node ./test-vad-with-non-streaming-asr-whisper.js
rm Obama.wav
rm silero_vad.onnx
rm -rf sherpa-onnx-moonshine-*

echo '-----vad+whisper----------'

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx
node ./test-vad-with-non-streaming-asr-whisper.js
rm Obama.wav
rm silero_vad.onnx
rm -rf sherpa-onnx-whisper-tiny.en

# offline asr
#
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

node ./test-offline-sense-voice.js

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
tar xf dict.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

node ./test-offline-sense-voice-with-hr.js

rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17
rm -rf dict replace.fst test-hr.wav lexicon.txt

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
ls -lh
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

rm -f itn*
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
node ./test-offline-paraformer-itn.js
rm -rf sherpa-onnx-paraformer-zh-2023-09-14

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
ls -lh
tar xvf sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
rm sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
node ./test-offline-nemo-ctc.js
rm -rf sherpa-onnx-nemo-ctc-en-conformer-small

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
ls -lh
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
node ./test-offline-paraformer.js
rm -rf sherpa-onnx-paraformer-zh-2023-09-14

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
ls -lh
tar xvf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
rm sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
node ./test-offline-transducer.js
rm -rf sherpa-onnx-zipformer-en-2023-06-26

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2
node ./test-offline-whisper.js
rm -rf sherpa-onnx-whisper-tiny.en

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

node ./test-offline-moonshine.js
rm -rf sherpa-onnx-moonshine-*
