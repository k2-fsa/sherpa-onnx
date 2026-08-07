# offline-punctuation

This example demonstrates how to use offline punctuation restoration in Flutter with sherpa-onnx.

It uses the [sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8](https://k2-fsa.github.io/sherpa/onnx/punctuation/pretrained_models.html#sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8) model,
which supports both Chinese and English text.

It works on the following platforms:

  - Android
  - iOS
  - Linux
  - macOS (both arm64 and x86_64 are supported)
  - Windows
  - Web

## Example

Input text (61 words):

```
yesterday afternoon 我去了一家 near my apartment 的 coffee shop 想要
enjoy a cup of hot latte 并 check 一些 important work emails 顺便在
我的 laptop 上写一点 code 因为那里的 atmosphere 总是非常 quiet and
comfortable 可以让我更加 focus on coding 和 writing documents 而且
没有任何 distractions 所以 if you also like this kind of relaxing
weekend vibe 我们 definitely 应该 plan 一个 time 一起 hang out.
```

Output text:

```
yesterday afternoon，我去了一家near my apartment的coffee shop，想要
enjoy a cup of hot latte，并check一些important work emails，顺便在
我的laptop上写一点code，因为那里的atmosphere总是非常quiet and
comfortable，可以让我更加focus on coding和writing documents，而且
没有任何distractions。所以if you also like this kind of relaxing
weekend vibe，我们definitely应该plan一个time一起hang out。
```

Performance (web demo, 1 thread): `Words: 61 | Elapsed: 0.035s`

## How to build

### 1. Download the model

```bash
cd flutter-examples/offline-punctuation/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8.tar.bz2
cd ..

./generate-asset-list.py
```

Note: `generate-asset-list.py` is a symlink to `../tts/generate-asset-list.py`.

### 2. Build the APP

  - For Linux

```bash
flutter build linux
```

  - For macOS

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

  - For Windows

```bash
flutter build windows
```

  - For Android

```bash
flutter build apk --split-per-abi
```

  - For web

```bash
flutter run -d chrome
```

Or build and serve:

```bash
flutter build web
cd build/web
python3 -m http.server 6006
```

Then open <http://localhost:6006> in your browser.

  - For iOS

Connect your iPhone, then:

```bash
flutter devices
flutter run -d <device-id> --release
```

If you get signing errors, open `ios/Runner.xcworkspace` in Xcode, configure your signing team, then retry.
