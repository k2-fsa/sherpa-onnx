#!/usr/bin/env bash

set -ex
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"
d=nodejs-addon-examples
echo "dir: $d"
cd $d

arch=$(node -p "require('os').arch()")
platform=$(node -p "require('os').platform()")
node_version=$(node -p "process.versions.node.split('.')[0]")

echo "----------Cohere Transcribe----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01.tar.bz2

node ./test_asr_non_streaming_cohere_transcribe.js
node ./test_asr_non_streaming_cohere_transcribe_async.js

rm -rf sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01

echo "----------Qwen3 ASR----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25.tar.bz2

node ./test_asr_non_streaming_qwen3_asr.js
node ./test_asr_non_streaming_qwen3_asr_async.js

rm -rf sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25

echo "----------Moonshine v2----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27.tar.bz2

node ./test_asr_non_streaming_moonshine_v2.js

rm -rf sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27

echo "----------FireRedAsr CTC----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25.tar.bz2

node ./test_asr_non_streaming_fire_red_asr_ctc.js
node ./test_asr_non_streaming_fire_red_asr_ctc_async.js

rm -rf sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25

echo "----------PocketTTS----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2

node ./test_tts_non_streaming_pocket_en.js
node ./test_tts_non_streaming_pocket_en_async.js

rm -rf sherpa-onnx-pocket-tts-int8-2026-01-26

echo "----------ZipVoice----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-zipvoice-distill-int8-zh-en-emilia.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos_24khz.onnx

node ./test_tts_non_streaming_zipvoice_zh_en.js
node ./test_tts_non_streaming_zipvoice_zh_en_async.js

rm -rf sherpa-onnx-zipvoice-distill-int8-zh-en-emilia
rm -f vocos_24khz.onnx

echo "----------non-streaming ASR FunASR Nano----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-funasr-nano-int8-2025-12-30.tar.bz2

node ./test_asr_non_streaming_funasr_nano.js
node ./test_asr_non_streaming_funasr_nano_async.js

rm -rf sherpa-onnx-funasr-nano-int8-2025-12-30

echo "----------non-streaming ASR Google MedASR CTC----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2

node ./test_asr_non_streaming_medasr_ctc.js

rm -rf sherpa-onnx-medasr-ctc-en-int8-2025-12-25

echo "----------non-streaming ASR Omnilingual ASR CTC----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2

node ./test_asr_non_streaming_omnilingual_asr_ctc.js

rm -rf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12

echo "----------non-streaming ASR WeNet CTC----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2

node ./test_asr_non_streaming_wenet_ctc.js
rm -rf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10

echo "----------streaming ASR T-one CTC----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2

node ./test_asr_streaming_t_one_ctc.js

rm -rf sherpa-onnx-streaming-t-one-russian-2025-09-08

echo "----------KittenTTS----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2

node ./test_tts_non_streaming_kitten_en.js
node ./test_tts_non_streaming_kitten_en_sync.js

rm -rf kitten-nano-en-v0_1-fp16

echo "----------SupertonicTTS----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2

node ./test_tts_non_streaming_supertonic_en.js
node ./test_tts_non_streaming_supertonic_en_async.js

rm -rf sherpa-onnx-supertonic-3-tts-int8-2026-05-11

echo "----------non-streaming ASR NeMo Canary----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2

node ./test_asr_non_streaming_nemo_canary.js

rm -rf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8

echo "----------non-streaming ASR Zipformer CTC----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

node ./test_asr_non_streaming_zipformer_ctc.js
rm -rf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03

echo "----------non-streaming ASR NeMo parakeet tdt----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2

node ./test_asr_non_streaming_nemo_parakeet_tdt_v2.js
node ./test_asr_non_streaming_nemo_parakeet_tdt_v2_hotwords.js
rm -rf sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8

echo "----------non-streaming ASR dolphin CTC----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2

node ./test_asr_non_streaming_dolphin_ctc.js

rm -rf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02

echo "----------non-streaming speech denoiser----------"

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

node ./test_offline_speech_enhancement_gtcrn.js
node ./test_offline_speech_enhancement_dpdfnet.js
node ./test_online_speech_enhancement_gtcrn.js
node ./test_online_speech_enhancement_dpdfnet.js
rm gtcrn_simple.onnx
rm dpdfnet_baseline.onnx
ls -lh *.wav

echo "----------non-streaming asr FireRedAsr----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2

node ./test_asr_non_streaming_fire_red_asr.js
rm -rf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16

echo "----------non-streaming asr moonshine + vad----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test_vad_with_non_streaming_asr_moonshine.js
rm -rf sherpa-onnx-*
rm *.wav
rm *.onnx

echo "----------non-streaming speaker diarization----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test_offline_speaker_diarization.js

rm -rfv *.onnx *.wav sherpa-onnx-pyannote-*

echo "----------non-streaming asr whisper + vad----------"
download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test_vad_with_non_streaming_asr_whisper.js
rm -rf sherpa-onnx-whisper*
rm *.wav
rm *.onnx

echo "----------asr----------"

if [[ $arch != "ia32" && $platform != "win32" ]]; then
  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

  node ./test_asr_non_streaming_nemo_ctc.js
  rm -rf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k

  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17.tar.bz2

  node ./test_asr_non_streaming_sense_voice.js

  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2

  download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

  node ./test_asr_non_streaming_sense_voice_with_hr.js

  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

  # node ./test_vad_asr_non_streaming_sense_voice_microphone.js

  rm -rf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17
  rm -rf dict replace.fst test-hr.wav lexicon.txt silero_vad.onnx

  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

  node ./test_asr_non_streaming_paraformer.js

  rm -f itn*

  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
  download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

  node ./test_asr_non_streaming_paraformer_itn.js

  rm -rf sherpa-onnx-paraformer-zh-2023-09-14
fi

echo "----------tts----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2

node ./test_tts_non_streaming_kokoro_zh_en.js
node ./test_tts_non_streaming_kokoro_zh_en_async.js
ls -lh *.wav
rm -rf kokoro-multi-lang-v1_0

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2

node ./test_tts_non_streaming_kokoro_en.js
node ./test_tts_non_streaming_kokoro_en_async.js
ls -lh *.wav
rm -rf kokoro-en-v0_19

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
download https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test_tts_non_streaming_matcha_icefall_en.js
node ./test_tts_non_streaming_matcha_icefall_en_async.js
rm vocos-22khz-univ.onnx
rm -rf matcha-icefall-en_US-ljspeech

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
download https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test_tts_non_streaming_matcha_icefall_zh.js
node ./test_tts_non_streaming_matcha_icefall_zh_async.js
rm vocos-22khz-univ.onnx
rm -rf matcha-icefall-zh-baker
ls -lh *.wav

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2

node ./test_tts_non_streaming_vits_piper_en.js
node ./test_tts_non_streaming_vits_piper_en_async.js
node ./test_tts_async_callback_stress.js ./vits-piper-en_GB-cori-medium 30
rm -rf vits-piper-en_GB-cori-medium

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2

node ./test_tts_non_streaming_vits_coqui_de.js
node ./test_tts_non_streaming_vits_coqui_de_async.js
rm -rf vits-coqui-de-css10

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2

node ./test_tts_non_streaming_vits_zh_ll.js
node ./test_tts_non_streaming_vits_zh_ll_async.js
rm -rf sherpa-onnx-vits-zh-ll

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2

node ./test_tts_non_streaming_vits_zh_aishell3.js
node ./test_tts_non_streaming_vits_zh_aishell3_async.js
rm -rf vits-icefall-zh-aishell3

echo "----------keyword spotting----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

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

  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

  node ./test_offline_punctuation.js
  rm -rf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12


  download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2

  node ./test_online_punctuation.js
fi

echo "----------audio tagging----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

node ./test_audio_tagging_zipformer.js
rm -rf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-04-19.tar.bz2

node ./test_audio_tagging_ced.js
rm -rf sherpa-onnx-ced-mini-audio-tagging-2024-04-19

echo "----------speaker identification----------"
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

git clone https://github.com/csukuangfj/sr-data

node ./test_speaker_identification.js

rm *.onnx
rm -rf sr-data

echo "----------spoken language identification----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2

node ./test_spoken_language_identification.js
rm -rf sherpa-onnx-whisper-tiny
rm -rf spoken-language-identification-test-wavs

echo "----------streaming asr----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

rm -f itn*

download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
download https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn-zh-number.wav

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
download https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

if [[ $arch != "ia32" && $platform != "win32" ]]; then
  node test_asr_streaming_transducer_itn.js
  node test_asr_streaming_transducer.js
  node test_asr_streaming_transducer_with_hr.js
fi

rm -rf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
rm -rf dict lexicon.txt replace.fst test-hr.wav

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

node ./test_asr_streaming_ctc.js

# To decode with HLG.fst
node ./test_asr_streaming_ctc_hlg.js
rm -rf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

node ./test_asr_streaming_paraformer.js
rm -rf sherpa-onnx-streaming-paraformer-bilingual-zh-en

echo "----------streaming ASR Nemotron multilingual----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11.tar.bz2

node ./test_asr_streaming_nemotron.js

rm -rf sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-1120ms-int8-2026-06-11

echo "----------non-streaming asr----------"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2

node ./test_asr_non_streaming_transducer.js
rm -rf sherpa-onnx-zipformer-en-2023-04-01

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2

node ./test_asr_non_streaming_whisper.js
rm -rf sherpa-onnx-whisper-tiny.en

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

node ./test_asr_non_streaming_moonshine.js
rm -rf sherpa-onnx-*

ls -lh
