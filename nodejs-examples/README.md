# Introduction

Note: You need `Node >= 18`.

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

## ./test-offline-tts-en.js

[./test-offline-tts-en.js](./test-offline-tts-en.js) shows how to use
[vits-piper-en_US-amy-low.tar.bz2](https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2)
for text-to-speech.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xvf vits-piper-en_US-amy-low.tar.bz2
node ./test-offline-tts-en.js
```

## ./test-offline-tts-zh.js

[./test-offline-tts-zh.js](./test-offline-tts-zh.js) shows how to use
a VITS pretrained model
[aishell3](https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#vits-model-aishell3)
for text-to-speech.

You can use the following command to run it:

```bash
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2
node ./test-offline-tts-zh.js
```

# Speech-to-text

In the following, we demonstrate how to decode files and how to perform
speech recognition with a microphone with `nodejs`.

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

## ./test-offline-sense-voice.js

[./test-offline-sense-voice.js](./test-offline-sense-voice.js) demonstrates
how to decode a file with a non-streaming Paraformer model.

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
