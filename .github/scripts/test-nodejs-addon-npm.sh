#!/usr/bin/env bash

set -ex

d=nodejs-addon-examples
echo "dir: $d"
cd $d

arch=$(node -p "require('os').arch()")
platform=$(node -p "require('os').platform()")
node_version=$(node -p "process.versions.node.split('.')[0]")

echo "----------non-streaming asr moonshine + vad----------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test_vad_with_non_streaming_asr_moonshine.js
rm -rf sherpa-onnx-*
rm *.wav
rm *.onnx

echo "----------non-streaming speaker diarization----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test_offline_speaker_diarization.js

rm -rfv *.onnx *.wav sherpa-onnx-pyannote-*

echo "----------non-streaming asr whisper + vad----------"
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test_vad_with_non_streaming_asr_whisper.js
rm -rf sherpa-onnx-whisper*
rm *.wav
rm *.onnx

echo "----------asr----------"

if [[ $arch != "ia32" && $platform != "win32" ]]; then
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
  tar xvf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
  rm sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

  node ./test_asr_non_streaming_nemo_ctc.js
  rm -rf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
  rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

  node ./test_asr_non_streaming_sense_voice.js
  rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
  rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  node ./test_asr_non_streaming_paraformer.js

  rm -f itn*

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

  node ./test_asr_non_streaming_paraformer_itn.js

  rm -rf sherpa-onnx-paraformer-zh-2023-09-14
fi

echo "----------tts----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
tar xf vits-piper-en_GB-cori-medium.tar.bz2
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

echo "----------keyword spotting----------"

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

node ./test_keyword_spotter_transducer.js
rm -rf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01

if [[ $arch != "ia32" && $platform != "win32" && $node_version != 21 ]]; then
  # The punctuation model is so large that it cause memory allocation failure on windows x86
  # 2024-07-17 03:24:34.2388391 [E:onnxruntime:, inference_session.cc:1981
  # onnxruntime::InferenceSession::Initialize::<lambda_d603a8c74863bd6b58a1c7996295ed04>::operator ()]
  # Exception during initialization: bad allocation
  # Error: Process completed with exit code 127.
  #
  # Node 21 does not have such an issue
  echo "----------add punctuations----------"

  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

  node ./test_punctuation.js
  rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
fi

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

rm -f itn*

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

if [[ $arch != "ia32" && $platform != "win32" ]]; then
  node test_asr_streaming_transducer_itn.js
  node test_asr_streaming_transducer.js
fi

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

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
rm sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

node ./test_asr_non_streaming_moonshine.js
rm -rf sherpa-onnx-*

ls -lh
