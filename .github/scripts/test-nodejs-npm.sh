#!/usr/bin/env bash

set -ex

echo "dir: $d"
cd $d
npm install
git status
ls -lh
ls -lh node_modules

echo '-----speaker diarization----------'
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test-offline-speaker-diarization.js
rm -rfv *.wav *.onnx sherpa-onnx-pyannote-*

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

echo "----------keyword spotting----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

node ./test-keyword-spotter-transducer.js
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

# offline asr
#
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

node ./test-offline-sense-voice.js
rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

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

# offline tts

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2
node ./test-offline-tts-en.js
rm -rf vits-piper-en_US-amy-low*

curl -LS -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
node ./test-offline-tts-zh.js
rm -rf vits-icefall-zh-aishell3*
