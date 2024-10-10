# Introduction

Note: You need `Node >= 16`.

This repo contains examples for NodeJS.
It uses [node-addon-api](https://github.com/nodejs/node-addon-api) to wrap
`sherpa-onnx` for NodeJS and it supports multiple threads.

Note: [../nodejs-examples](../nodejs-examples) uses WebAssembly to wrap
`sherpa-onnx` for NodeJS and it does not support multiple threads.

Before you continue, please first run

```bash
npm install # or pnpm install

# For macOS x64
## With npm
export DYLD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-darwin-x64:$DYLD_LIBRARY_PATH
## With pnpm
export DYLD_LIBRARY_PATH=$PWD/node_modules/.pnpm/sherpa-onnx-node@<REPLACE-THIS-WITH-THE-INSTALLED-VERSION>/node_modules/sherpa-onnx-darwin-x64:$DYLD_LIBRARY_PATH

# For macOS arm64
## With npm
export DYLD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-darwin-arm64:$DYLD_LIBRARY_PATH
## With pnpm
export DYLD_LIBRARY_PATH=$PWD/node_modules/.pnpm/sherpa-onnx-node@<REPLACE-THIS-WITH-THE-INSTALLED-VERSION>/node_modules/sherpa-onnx-darwin-arm64:$DYLD_LIBRARY_PATH

# For Linux x64
## With npm
export LD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-linux-x64:$LD_LIBRARY_PATH
## With pnpm
export LD_LIBRARY_PATH=$PWD/node_modules/.pnpm/sherpa-onnx-node@<REPLACE-THIS-WITH-THE-INSTALLED-VERSION>/node_modules/sherpa-onnx-linux-x64:$LD_LIBRARY_PATH

# For Linux arm64, e.g., Raspberry Pi 4
## With npm
export LD_LIBRARY_PATH=$PWD/node_modules/sherpa-onnx-linux-arm64:$LD_LIBRARY_PATH
## With pnpm
export LD_LIBRARY_PATH=$PWD/node_modules/.pnpm/sherpa-onnx-node@<REPLACE-THIS-WITH-THE-INSTALLED-VERSION>/node_modules/sherpa-onnx-linux-arm64:$LD_LIBRARY_PATH
```

# Examples

The following tables list the examples in this folder.

## Speaker diarization

|File| Description|
|---|---|
|[./test_offline_speaker_diarization.js](./test_offline_speaker_diarization.js)| It demonstrates how to use sherpa-onnx JavaScript API for speaker diarization. It supports speaker segmentation models from [pyannote-audio](https://github.com/pyannote/pyannote-audio)|

## Add punctuations to text

|File| Description|
|---|---|
|[./test_punctuation.js](./test_punctuation.js)| Add punctuations to input text using [CT transformer](https://modelscope.cn/models/iic/punc_ct-transformer_zh-cn-common-vocab272727-pytorch/summary). It supports both Chinese and English.|

## Voice activity detection (VAD)

|File| Description|
|---|---|
|[./test_vad_microphone.js](./test_vad_microphone.js)| VAD with a microphone. It uses [silero-vad](https://github.com/snakers4/silero-vad)|

## Speaker identification

|File| Description|
|---|---|
|[ ./test_speaker_identification.js]( ./test_speaker_identification.js)| Speaker identification from a file|

## Spoken language identification

|File| Description|
|---|---|
|[./test_vad_spoken_language_identification_microphone.js](./test_vad_spoken_language_identification_microphone.js)|Spoken language identification from a microphone using a multi-lingual [Whisper](https://github.com/openai/whisper) model|

## Audio tagging

|File| Description|
|---|---|
|[./test_audio_tagging_zipformer.js](./test_audio_tagging_zipformer.js)| Audio tagging with a Zipformer model|
|[./test_audio_tagging_ced.js](./test_audio_tagging_ced.js)| Audio tagging with a [CED](https://github.com/RicherMans/CED) model|

## Keyword spotting

|File| Description|
|---|---|
|[./test_keyword_spotter_transducer.js](./test_keyword_spotter_transducer.js)| Keyword spotting from a file using a Zipformer model|
|[./test_keyword_spotter_transducer_microphone.js](./test_keyword_spotter_transducer_microphone.js)| Keyword spotting from a microphone using a Zipformer model|

## Streaming speech-to-text from files

|File| Description|
|---|---|
|[./test_asr_streaming_transducer.js](./test_asr_streaming_transducer.js)| Streaming speech recognition from a file using a Zipformer transducer model|
|[./test_asr_streaming_ctc.js](./test_asr_streaming_ctc.js)| Streaming speech recognition from a file using a Zipformer CTC model with greedy search|
|[./test_asr_streaming_ctc_hlg.js](./test_asr_streaming_ctc_hlg.js)| Streaming speech recognition from a file using a Zipformer CTC model with HLG decoding|
|[./test_asr_streaming_paraformer.js](./test_asr_streaming_paraformer.js)|Streaming speech recognition from a file using a [Paraformer](https://github.com/alibaba-damo-academy/FunASR) model|

## Streaming speech-to-text from a microphone

|File| Description|
|---|---|
|[./test_asr_streaming_transducer_microphone.js](./test_asr_streaming_transducer_microphone.js)| Streaming speech recognition from a microphone using a Zipformer transducer model|
|[./test_asr_streaming_ctc_microphone.js](./test_asr_streaming_ctc_microphone.js)| Streaming speech recognition from a microphone using a Zipformer CTC model with greedy search|
|[./test_asr_streaming_ctc_hlg_microphone.js](./test_asr_streaming_ctc_hlg_microphone.js)|Streaming speech recognition from a microphone using a Zipformer CTC model with HLG decoding|
|[./test_asr_streaming_paraformer_microphone.js](./test_asr_streaming_paraformer_microphone.js)| Streaming speech recognition from a microphone using a [Paraformer](https://github.com/alibaba-damo-academy/FunASR) model|

## Non-Streaming speech-to-text from files

|File| Description|
|---|---|
|[./test_asr_non_streaming_transducer.js](./test_asr_non_streaming_transducer.js)|Non-streaming speech recognition from a file with a Zipformer transducer model|
|[./test_asr_non_streaming_whisper.js](./test_asr_non_streaming_whisper.js)| Non-streaming speech recognition from a file using [Whisper](https://github.com/openai/whisper)|
|[./test_vad_with_non_streaming_asr_whisper.js](./test_vad_with_non_streaming_asr_whisper.js)| Non-streaming speech recognition from a file using [Whisper](https://github.com/openai/whisper) + [Silero VAD](https://github.com/snakers4/silero-vad)|
|[./test_asr_non_streaming_nemo_ctc.js](./test_asr_non_streaming_nemo_ctc.js)|Non-streaming speech recognition from a file using a [NeMo](https://github.com/NVIDIA/NeMo) CTC model with greedy search|
|[./test_asr_non_streaming_paraformer.js](./test_asr_non_streaming_paraformer.js)|Non-streaming speech recognition from a file using [Paraformer](https://github.com/alibaba-damo-academy/FunASR)|
|[./test_asr_non_streaming_sense_voice.js](./test_asr_non_streaming_sense_voice.js)|Non-streaming speech recognition from a file using [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)|

## Non-Streaming speech-to-text from a microphone with VAD

|File| Description|
|---|---|
|[./test_vad_asr_non_streaming_transducer_microphone.js](./test_vad_asr_non_streaming_transducer_microphone.js)|VAD + Non-streaming speech recognition from a microphone using a Zipformer transducer model|
|[./test_vad_asr_non_streaming_whisper_microphone.js](./test_vad_asr_non_streaming_whisper_microphone.js)|VAD + Non-streaming speech recognition from a microphone using [Whisper](https://github.com/openai/whisper)|
|[./test_vad_asr_non_streaming_nemo_ctc_microphone.js](./test_vad_asr_non_streaming_nemo_ctc_microphone.js)|VAD + Non-streaming speech recognition from a microphone using a [NeMo](https://github.com/NVIDIA/NeMo) CTC model with greedy search|
|[./test_vad_asr_non_streaming_paraformer_microphone.js](./test_vad_asr_non_streaming_paraformer_microphone.js)|VAD + Non-streaming speech recognition from a microphone using [Paraformer](https://github.com/alibaba-damo-academy/FunASR)|
|[./test_vad_asr_non_streaming_sense_voice_microphone.js](./test_vad_asr_non_streaming_sense_voice_microphone.js)|VAD + Non-streaming speech recognition from a microphone using [SenseVoice](https://github.com/FunAudioLLM/SenseVoice)|

## Text-to-speech

|File| Description|
|---|---|
|[./test_tts_non_streaming_vits_piper_en.js](./test_tts_non_streaming_vits_piper_en.js)| Text-to-speech with a [piper](https://github.com/rhasspy/piper) English model|
|[./test_tts_non_streaming_vits_coqui_de.js](./test_tts_non_streaming_vits_coqui_de.js)| Text-to-speech with a [coqui](https://github.com/coqui-ai/TTS) German model|
|[./test_tts_non_streaming_vits_zh_ll.js](./test_tts_non_streaming_vits_zh_ll.js)| Text-to-speech with a Chinese model using [cppjieba](https://github.com/yanyiwu/cppjieba)|
|[./test_tts_non_streaming_vits_zh_aishell3.js](./test_tts_non_streaming_vits_zh_aishell3.js)| Text-to-speech with a Chinese TTS model|


### Speaker diarization

```bash

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test_offline_speaker_diarization.js
```

### Voice Activity detection (VAD)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx


# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_vad_microphone.js
```

### Audio tagging with zipformer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
tar xvf sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2
rm sherpa-onnx-zipformer-small-audio-tagging-2024-04-15.tar.bz2

node ./test_audio_tagging_zipformer.js
```

### Audio tagging with CED

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-ced-mini-audio-tagging-2024-09-14.tar.bz2
tar xvf sherpa-onnx-ced-mini-audio-tagging-2024-09-14.tar.bz2
rm sherpa-onnx-ced-mini-audio-tagging-2024-09-14.tar.bz2

node ./test_audio_tagging_ced.js
```

### Streaming speech recognition with Zipformer transducer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
rm sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2

node ./test_asr_streaming_transducer.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_transducer_microphone.js
```

### Streaming speech recognition with Zipformer CTC

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2

node ./test_asr_streaming_ctc.js

# To decode with HLG.fst
node ./test_asr_streaming_ctc_hlg.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_ctc_microphone.js
node ./test_asr_streaming_ctc_hlg_microphone.js
```

### Streaming speech recognition with Paraformer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xvf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2

node ./test_asr_streaming_paraformer.js

# To run the test with a microphone, you need to install the package naudiodon2
npm install naudiodon2

node ./test_asr_streaming_paraformer_microphone.js
```

### Non-streaming speech recognition with Zipformer transducer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
tar xvf sherpa-onnx-zipformer-en-2023-04-01.tar.bz2
rm sherpa-onnx-zipformer-en-2023-04-01.tar.bz2

node ./test_asr_non_streaming_transducer.js

# To run VAD + non-streaming ASR with transudcer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_transducer_microphone.js
```

### Non-streaming speech recognition with Whisper

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

node ./test_asr_non_streaming_whisper.js

# To run VAD + non-streaming ASR with Whisper using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_whisper_microphone.js
```

### Non-streaming speech recognition with Whisper + VAD

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test_vad_with_non_streaming_asr_whisper.js
```

### Non-streaming speech recognition with NeMo CTC models

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
tar xvf sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2
rm sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k.tar.bz2

node ./test_asr_non_streaming_nemo_ctc.js

# To run VAD + non-streaming ASR with Paraformer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_nemo_ctc_microphone.js
```

### Non-streaming speech recognition with Paraformer

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
rm sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2

node ./test_asr_non_streaming_paraformer.js

# To run VAD + non-streaming ASR with Paraformer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_paraformer_microphone.js
```

### Non-streaming speech recognition with SenseVoice

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

node ./test_asr_non_streaming_sense_voice.js

# To run VAD + non-streaming ASR with Paraformer using a microphone
npm install naudiodon2
node ./test_vad_asr_non_streaming_sense_voice_microphone.js
```

### Text-to-speech with piper VITS models (TTS)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_GB-cori-medium.tar.bz2
tar xvf vits-piper-en_GB-cori-medium.tar.bz2
rm vits-piper-en_GB-cori-medium.tar.bz2

node ./test_tts_non_streaming_vits_piper_en.js
```

### Text-to-speech with piper Coqui-ai/TTS models (TTS)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
tar xvf vits-coqui-de-css10.tar.bz2
rm vits-coqui-de-css10.tar.bz2

node ./test_tts_non_streaming_vits_coqui_de.js
```

### Text-to-speech with vits Chinese models (1/2)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
tar xvf sherpa-onnx-vits-zh-ll.tar.bz2
rm sherpa-onnx-vits-zh-ll.tar.bz2

node ./test_tts_non_streaming_vits_zh_ll.js
```

### Text-to-speech with vits Chinese models (2/2)

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
rm vits-icefall-zh-aishell3.tar.bz2

node ./test_tts_non_streaming_vits_zh_aishell3.js
```

### Spoken language identification with Whisper multi-lingual models

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.tar.bz2
rm sherpa-onnx-whisper-tiny.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
tar xvf spoken-language-identification-test-wavs.tar.bz2
rm spoken-language-identification-test-wavs.tar.bz2

node ./test_spoken_language_identification.js

# To run VAD + spoken language identification using a microphone
npm install naudiodon2
node ./test_vad_spoken_language_identification_microphone.js
```

### Speaker identification

You can find more models at
<https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models>

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

git clone https://github.com/csukuangfj/sr-data

node ./test_speaker_identification.js
```

### Add punctuations

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2

node ./test_punctuation.js
```

## Keyword spotting

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
rm sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2

node ./test_keyword_spotter_transducer.js

# To run keyword spotting using a microphone
npm install naudiodon2
node ./test_keyword_spotter_transducer_microphone.js
```
