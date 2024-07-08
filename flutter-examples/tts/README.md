# tts

This example demonstrates how to use text to speech (TTS) in Flutter with sherpa-onnx.

It works on the following platforms:

  - Android
  - iOS
  - Linux
  - macOS (both arm64 and x86_64 are supported)
  - Windows

## How to build

Before you run `flutter build`, you have to select a TTS model and change
the code to use your selected model.

### 1. Select a TTS model

We have a list of TTS models at

<https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models>

You can select any of them. If you feel that there are so many that you don't know
which one is the best, please visit <http://huggingface.co/spaces/k2-fsa/text-to-speech>
and try each one by yourself and select the one you consider the best.

Suppose you select

  <https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2>

Then please do the following:

  - 1. Download and unzip the model

```bash
cd flutter-examples/tts/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
tar xf vits-piper-en_US-libritts_r-medium.tar.bz2
rm vits-piper-en_US-libritts_r-medium.tar.bz2
cd ..

./generate-asset-list.py
```

  Note that you have to run [./generate-asset-list.py](./generate-asset-list.py) so that Flutter knows where
  to find the model.

  - 2. Change the code to use the downloaded model.

    We have given several examples for different models in [./lib/model.dart](./lib/model.dart).
    For our selected model, we need to change [./lib/model.dart](./lib/model.dart) so that it looks like below:

```
// Example 6
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
modelDir = 'vits-piper-en_US-libritts_r-medium';
modelName = 'en_US-libritts_r-medium.onnx';
dataDir = 'vits-piper-en_US-libritts_r-medium/espeak-ng-data';
```

  - 3. That's it.

### Build the APP

  - 1. For Linux

```bash
flutter build linux

# See below if you get any errors
```

  - 2. For macOS

To build a universal2 APP, use

```bash
flutter build macos
```

To build for `x86_64`, use

```bash
export FLUTTER_XCODE_ARCHS=x86_64
flutter build macos
```

To build for `arm64`, use

```bash
export FLUTTER_XCODE_ARCHS=arm64
flutter build macos
```

  - 3. For Windows

```bash
flutter build windows
```

  - 4. For Android

```bash
flutter build apk --split-per-abi
```

  - 5. For iOS

```
flutter build ios
```

## Fix for Linux

If you get the following errors on Linux,

```
Building Linux application...
CMake Error at /usr/local/share/cmake-3.29/Modules/FindPkgConfig.cmake:634 (message):
  The following required packages were not found:

   - gstreamer-1.0

Call Stack (most recent call first):
  /usr/local/share/cmake-3.29/Modules/FindPkgConfig.cmake:862 (_pkg_check_modules_internal)
  flutter/ephemeral/.plugin_symlinks/audioplayers_linux/linux/CMakeLists.txt:24 (pkg_check_modules)
```

please run:

```bash
sudo apt-get install -y libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libunwind-dev
```

See also <https://github.com/bluefireteam/audioplayers/tree/main/packages/audioplayers_linux#setup-for-linux>
for the above error.
