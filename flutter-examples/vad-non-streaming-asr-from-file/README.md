# vad-non-streaming-asr-from-file

This example demonstrates how to use Voice Activity Detection (VAD) with
non-streaming speech recognition from an audio file in Flutter with sherpa-onnx.

It uses:
- [Silero VAD v4](https://k2-fsa.github.io/sherpa/onnx/vad/silero-vad.html) for voice activity detection
- [Zipformer CTC](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/icefall/zipformer.html) for speech recognition (default model)

Other supported ASR models: SenseVoice, Whisper. Change `selectedModelIndex`
in `lib/model_config.dart` to switch models.

It works on the following platforms:

  - Android
  - iOS
  - Linux
  - macOS (both arm64 and x86_64 are supported)
  - Windows
  - Web

## Features

- Configurable VAD parameters (threshold, min silence, min speech, max speech)
- Progress bar showing processing status
- Displays elapsed time, audio duration, and RTF after processing
- Displays ASR transcription for each detected segment
- Click a segment to play it back

## How to build

### 1. Download the models

Download the VAD model and an ASR model.

**Default: Zipformer CTC (Chinese)**

```bash
cd flutter-examples/vad-non-streaming-asr-from-file/assets
rm -f .gitkeep

# VAD model
curl -SL -o silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# ASR model (Zipformer CTC)
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
tar xvf sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
rm sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2

# Remove files not required by the app (test wavs, vocab, etc.)
cd sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03
rm -f *.wav README.md bbpe.model bbpe.vocab
rm -rf test_wavs
cd ..

cd ..
./generate-asset-list.py
```

After downloading, the `assets/` directory should contain only:

```
assets/
├── silero_vad.onnx
└── sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/
    ├── model.int8.onnx
    └── tokens.txt
```

**Alternative: Whisper tiny.en (English)**

To use Whisper instead of Zipformer CTC, download the Whisper model and
change `selectedModelIndex` to `2` in `lib/model_config.dart`:

```bash
cd flutter-examples/vad-non-streaming-asr-from-file/assets
rm -f .gitkeep

# VAD model
curl -SL -o silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# ASR model (Whisper tiny.en)
curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
rm sherpa-onnx-whisper-tiny.en.tar.bz2

cd ..
./generate-asset-list.py
```

Then edit `lib/model_config.dart` and set `const int selectedModelIndex = 2;`.

Note: `generate-asset-list.py` is a symlink to `../tts/generate-asset-list.py`.

### 2. Build the APP

  - For Linux

Install `libmpv-dev` first (required by the `media_kit` audio/video player):

```bash
sudo apt-get install -y libmpv-dev
flutter build linux
```

  - For macOS

```bash
flutter build macos
```

  - For Windows

```bash
flutter build windows
```

  - For Android

```bash
flutter build apk --split-per-abi --target-platform android-arm64
```

  - For web

```bash
flutter run -d chrome
```

  - For iOS

Connect your iPhone, then:

```bash
flutter devices
flutter run -d <device-id> --release
```
