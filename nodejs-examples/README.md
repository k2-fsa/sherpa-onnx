# Introduction

Note: You need `Node >= 18`. 

Note: For Mac M1 and other silicon chip series, do check the example `test-online-paraformer-microphone-mic.js` 

This directory contains nodejs examples for [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).

It uses WebAssembly to wrap `sherpa-onnx` for NodeJS and it does not support multiple threads.

Note: [../nodejs-addon-examples](../nodejs-addon-examples) uses
[node-addon-api](https://github.com/nodejs/node-addon-api) to wrap
`sherpa-onnx` for NodeJS and it supports multiple threads.

Before you continue, please first run

```bash
cd ./nodejs-examples

npm i
```

In the following, we describe how to use [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
for text-to-speech and speech-to-text.


# Speech enhancement

In the following, we demonstrate how to run speech enhancement.

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
node ./test-offline-speech-enhancement-gtcrn.js
```

# Speaker diarization

In the following, we demonstrate how to run speaker diarization.

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

node ./test-offline-speaker-diarization.js
```

# Text-to-speech

In the following, we demonstrate how to run text-to-speech.

## ./test-offline-tts-kitten-en.js

[./test-offline-tts-kitten-en.js](./test-offline-tts-kitten-en.js) shows how to use
[kitten-nano-en-v0_1-fp16](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2)
for text-to-speech.

You can use the following command to run it:

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kitten-nano-en-v0_1-fp16.tar.bz2
tar xf kitten-nano-en-v0_1-fp16.tar.bz2
rm kitten-nano-en-v0_1-fp16.tar.bz2

node ./test-offline-tts-kitten-en.js
```

## ./test-offline-tts-kokoro-en.js

[./test-offline-tts-kokoro-en.js](./test-offline-tts-kokoro-en.js) shows how to use
[kokoro-en-v0_19](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2)
for text-to-speech.

You can use the following command to run it:

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2
tar xf kokoro-en-v0_19.tar.bz2
rm kokoro-en-v0_19.tar.bz2

node ./test-offline-tts-kokoro-en.js
```

## ./test-offline-tts-matcha-zh.js

[./test-offline-tts-matcha-zh.js](./test-offline-tts-matcha-zh.js) shows how to use
[matcha-icefall-zh-baker](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker)
for text-to-speech.

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test-offline-tts-matcha-zh.js
```

## ./test-offline-tts-matcha-en.js

[./test-offline-tts-matcha-en.js](./test-offline-tts-matcha-en.js) shows how to use
[matcha-icefall-en_US-ljspeech](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker)
for text-to-speech.

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/vocos-22khz-univ.onnx

node ./test-offline-tts-matcha-en.js
```

## ./test-offline-tts-vits-en.js

[./test-offline-tts-vits-en.js](./test-offline-tts-vits-en.js) shows how to use
[vits-piper-en_US-amy-low.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2)
for text-to-speech.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xvf vits-piper-en_US-amy-low.tar.bz2
node ./test-offline-tts-vits-en.js
```

## ./test-offline-tts-vits-zh.js

[./test-offline-tts-vits-zh.js](./test-offline-tts-vits-zh.js) shows how to use
a VITS pretrained model
[aishell3](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-model-aishell3)
for text-to-speech.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
node ./test-offline-tts-vits-zh.js
```

# Speech-to-text

In the following, we demonstrate how to decode files and how to perform
speech recognition with a microphone with `nodejs`.

## ./test-offline-dolphin-ctc.js

[./test-offline-dolphin-ctc.js](./test-offline-dolphin-ctc.js) demonstrates
how to decode a file with a [Dolphin](https://github.com/DataoceanAI/Dolphin) CTC model.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
tar xvf sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
rm sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02.tar.bz2
node ./test-offline-dolphin-ctc.js
```

## ./test-offline-nemo-canary.js

[./test-offline-nemo-canary.js](./test-offline-nemo-canary.js) demonstrates
how to decode a file with a NeMo Canary model. In the code we use
[sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8](https://k2-fsa.github.io/sherpa/onnx/nemo/canary.html#sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8-english-spanish-german-french).

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
tar xvf sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2
rm sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8.tar.bz2

node ./test-offline-nemo-canary.js
```

## ./test-offline-zipformer-ctc.js

[./test-offline-zipformer-ctc.js](./test-offline-zipformer-ctc.js) demonstrates
how to decode a file with a Zipformer CTC model. In the code we use
[sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html#sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03-chinese).

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

node ./test-offline-zipformer-ctc.js
```

## ./test-offline-medasr-ctc.js

[./test-offline-medasr-ctc.js](./test-offline-medasr-ctc.js) demonstrates
how to decode a file with a Google MedASR CTC model. In the code we use
[sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2).

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
tar xvf sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2
rm sherpa-onnx-medasr-ctc-en-int8-2025-12-25.tar.bz2

node ./test-offline-medasr-ctc.js
```

## ./test-offline-omnilingual-asr-ctc.js

[./test-offline-omnilingual-asr-ctc.js](./test-offline-omnilingual-asr-ctc.js) demonstrates
how to decode a file with a Omnilingual ASR CTC model. In the code we use
[sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2).

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
tar xvf sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2
rm sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12.tar.bz2

node ./test-offline-omnilingual-asr-ctc.js
```

## ./test-offline-wenet-ctc.js

[./test-offline-wenet-ctc.js](./test-offline-wenet-ctc.js) demonstrates
how to decode a file with a WeNet CTC model. In the code we use
[sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2).

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
tar xvf sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2
rm sherpa-onnx-wenetspeech-yue-u2pp-conformer-ctc-zh-en-cantonese-int8-2025-09-10.tar.bz2

node ./test-offline-wenet-ctc.js
```

## ./test-offline-nemo-ctc.js

[./test-offline-nemo-ctc.js](./test-offline-nemo-ctc.js) demonstrates
how to decode a file with a NeMo CTC model. In the code we use
[stt_en_conformer_ctc_small](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/nemo/english.html#stt-en-conformer-ctc-small).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
tar xvf sherpa-onnx-nemo-ctc-en-conformer-small.tar.bz2
node ./test-offline-nemo-ctc.js
```

## ./test-offline-paraformer.js

[./test-offline-paraformer.js](./test-offline-paraformer.js) demonstrates
how to decode a file with a non-streaming Paraformer model. In the code we use
[sherpa-onnx-paraformer-zh-2023-09-14](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-09-14-chinese).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
tar xvf sherpa-onnx-paraformer-zh-2023-09-14.tar.bz2
node ./test-offline-paraformer.js
```

## ./test-offline-sense-voice-with-hr.js

[./test-offline-sense-voice-with-hr.js](./test-offline-sense-voice-with-hr.js) demonstrates
how to decode a file with a non-streaming SenseVoice model with homophone replacer.

You can use the following command to run it:

```bash
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/dict.tar.bz2
tar xf dict.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/replace.fst
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/test-hr.wav
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/hr-files/lexicon.txt

node ./test-offline-sense-voice-with-hr.js
```

## ./test-offline-sense-voice.js

[./test-offline-sense-voice.js](./test-offline-sense-voice.js) demonstrates
how to decode a file with a non-streaming SenseVoice model.

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
tar xvf sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2
rm sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17.tar.bz2

node ./test-offline-sense-voice.js
```

## ./test-offline-transducer.js

[./test-offline-transducer.js](./test-offline-transducer.js) demonstrates
how to decode a file with a non-streaming transducer model. In the code we use
[sherpa-onnx-zipformer-en-2023-06-26](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-zipformer-en-2023-06-26-english).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
tar xvf sherpa-onnx-zipformer-en-2023-06-26.tar.bz2
node ./test-offline-transducer.js
```

## ./test-vad-with-non-streaming-asr-whisper.js

[./test-vad-with-non-streaming-asr-whisper.js](./test-vad-with-non-streaming-asr-whisper.js)
shows how to use VAD + whisper to decode a very long file.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test-vad-with-non-streaming-asr-whisper.js
```

## ./test-offline-whisper.js

[./test-offline-whisper.js](./test-offline-whisper.js) demonstrates
how to decode a file with a Whisper model. In the code we use
[sherpa-onnx-whisper-tiny.en](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
node ./test-offline-whisper.js
```

## ./test-offline-fire-red-asr.js

[./test-offline-fire-red-asr.js](./test-offline-fire-red-asr.js) demonstrates
how to decode a file with a FireRedAsr AED model.

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
tar xvf sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2
rm sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16.tar.bz2

node ./test-offline-fire-red-asr.js
```

## ./test-offline-moonshine.js

[./test-offline-moonshine.js](./test-offline-moonshine.js) demonstrates
how to decode a file with a Moonshine model. In the code we use
[sherpa-onnx-moonshine-tiny-en-int8](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2).

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

node ./test-offline-moonshine.js
```

## ./test-vad-with-non-streaming-asr-moonshine.js

[./test-vad-with-non-streaming-asr-moonshine.js](./test-vad-with-non-streaming-asr-moonshine.js)
shows how to use VAD + whisper to decode a very long file.

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-moonshine-tiny-en-int8.tar.bz2
tar xvf sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

node ./test-vad-with-non-streaming-asr-moonshine.js
```

## ./test-online-paraformer-microphone.js

[./test-online-paraformer-microphone.js](./test-online-paraformer-microphone.js)
demonstrates how to do real-time speech recognition from microphone
with a streaming Paraformer model. In the code we use
[sherpa-onnx-streaming-paraformer-bilingual-zh-en](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
node ./test-online-paraformer-microphone.js
```


## ./test-online-paraformer-microphone-mic.js

[./test-online-paraformer-microphone-mic.js](./test-online-paraformer-microphone-mic.js)
demonstrates how to do real-time speech recognition from microphone
with a streaming Paraformer model. In the code we use
[sherpa-onnx-streaming-paraformer-bilingual-zh-en](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english).

It uses `mic` for better compatibility, do check its [npm](https://www.npmjs.com/package/mic) before running it.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
node ./test-online-paraformer-microphone-mic.js
```

## ./test-online-t-one-ctc.js
[./test-online-t-one-ctc.js](./test-online-t-one-ctc.js) demonstrates
how to decode a file using a streaming T-one model.

You can use the following command to run it:

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
tar xvf sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
rm sherpa-onnx-streaming-t-one-russian-2025-09-08.tar.bz2
node ./test-online-t-one-ctc.js
```

## ./test-online-paraformer.js
[./test-online-paraformer.js](./test-online-paraformer.js) demonstrates
how to decode a file using a streaming Paraformer model. In the code we use
[sherpa-onnx-streaming-paraformer-bilingual-zh-en](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
rm sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
node ./test-online-paraformer.js
```

## ./test-online-transducer-microphone.js
[./test-online-transducer-microphone.js](./test-online-transducer-microphone.js)
demonstrates how to do real-time speech recognition with microphone using a streaming transducer model. In the code
we use [sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english).


You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
node ./test-online-transducer-microphone.js
```

## ./test-online-transducer.js
[./test-online-transducer.js](./test-online-transducer.js) demonstrates
how to decode a file using a streaming transducer model. In the code
we use [sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2
node ./test-online-transducer.js
```

## ./test-online-zipformer2-ctc.js
[./test-online-zipformer2-ctc.js](./test-online-zipformer2-ctc.js) demonstrates
how to decode a file using a streaming zipformer2 CTC model. In the code
we use [sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-ctc/zipformer-ctc-models.html#sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13-chinese).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
node ./test-online-zipformer2-ctc.js
```

## ./test-online-zipformer2-ctc-hlg.js
[./test-online-zipformer2-ctc-hlg.js](./test-online-zipformer2-ctc-hlg.js) demonstrates
how to decode a file using a streaming zipformer2 CTC model with HLG. In the code
we use [sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2).

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
node ./test-online-zipformer2-ctc-hlg.js
```
