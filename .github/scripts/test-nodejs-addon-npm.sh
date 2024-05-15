#!/usr/bin/env bash

set -ex

d=nodejs-addon-examples
echo "dir: $d"
cd $d

echo "----------keyword spotting----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

node ./test_keyword_spotter_transducer.js
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

echo "----------add punctuations----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

node ./test_punctuation.js
rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12

echo "----------audio tagging----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

node ./test_audio_tagging_zipformer.js
rm -rf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
tar xvf sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2
rm sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2

node ./test_audio_tagging_ced.js
rm -rf sherpa-onnx-ced-mini-audio-tagging-2024-04-19

echo "----------speaker identification----------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

git clone https://github.com/csukuangfj/sr-data

node ./test_speaker_identification.js

rm *.onnx
rm -rf sr-data

echo "----------spoken language identification----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
rm sherpa-onnx-whisper-tiny.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
tar xvf spoken-language-identification-test-wavs.tar.bz2
rm spoken-language-identification-test-wavs.tar.bz2

node ./test_spoken_language_identification.js
rm -rf sherpa-onnx-whisper-tiny
rm -rf spoken-language-identification-test-wavs

echo "----------streaming asr----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

node test_asr_streaming_transducer.js

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

node ./test_asr_streaming_ctc.js

# To decode with HLG.fst
node ./test_asr_streaming_ctc_hlg.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

node ./test_asr_streaming_paraformer.js
rm -rf sherpa-onnx-streaming-paraformer-bilingual-zh-en

echo "----------non-streaming asr----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2

node ./test_asr_non_streaming_transducer.js
rm -rf sherpa-onnx-zipformer-en-2023-04-01

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

node ./test_asr_non_streaming_whisper.js
rm -rf sherpa-onnx-whisper-tiny.en

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
tar xvf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
rm sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

node ./test_asr_non_streaming_nemo_ctc.js
rm -rf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2

node ./test_asr_non_streaming_paraformer.js
rm -rf sherpa-onnx-paraformer-zh-2023-03-28

echo "----------tts----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
tar xvf vits-piper-en_GB-cori-medium.tar.bz2
rm vits-piper-en_GB-cori-medium.tar.bz2

node ./test_tts_non_streaming_vits_piper_en.js
rm -rf vits-piper-en_GB-cori-medium

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
tar xvf vits-coqui-de-css10.tar.bz2
rm vits-coqui-de-css10.tar.bz2

node ./test_tts_non_streaming_vits_coqui_de.js
rm -rf vits-coqui-de-css10

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
tar xvf sherpa-onnx-vits-zh-ll.tar.bz2
rm sherpa-onnx-vits-zh-ll.tar.bz2

node ./test_tts_non_streaming_vits_zh_ll.js
rm -rf sherpa-onnx-vits-zh-ll

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
rm vits-icefall-zh-aishell3.tar.bz2

node ./test_tts_non_streaming_vits_zh_aishell3.js
rm -rf vits-icefall-zh-aishell3

ls -lh
