# Real-time speech recognition

This APP supports the following platforms:

  - macOS(so far)

## Getting Started

Remember to use the following steps to download a model. Otherwise, you would
get errors after you start and run the app.

###  1. Select a streaming model

Please visit <https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models>
to download a streaming ASR model.

You can find introductions about each streaming model at
<https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html>


So far the demo is made on whisper and SileroVad.

### 2. Place your downloaded model inside the directory assets

The downloaded model has to be placed in the [assets](./assets) directory.

**HINT**: Please delete files that are not needed by the code. Otherwise, you put
unnecessary files in your APP and it will significantly increase the size of your APP.

## Example

Suppose you have selected the following model

<https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2>

Download the model to the [./assets](./assets) directory.

Your [assets](./assets) directory should look like below at the end.

```
./assets
├── base-decoder.onnx
├── base-encoder.onnx
├── base-tokens.txt
└── silero_vad.onnx

0 directory, 4 files
```

  - 3. Run it!

    For instance
      - `flutter run -d macos` for macOS.
