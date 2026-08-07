# online-punctuation

This example demonstrates how to use online punctuation restoration in Flutter with sherpa-onnx.

It uses the [sherpa-onnx-online-punct-en-2024-08-06](https://k2-fsa.github.io/sherpa/onnx/punctuation/pretrained_models.html#sherpa-onnx-online-punct-en-2024-08-06) model,
which supports English text.

It works on the following platforms:

  - Android
  - iOS
  - Linux
  - macOS (both arm64 and x86_64 are supported)
  - Windows
  - Web

## Example

Input text (145 words):

```
last week my friends and i decided to take a spontaneous road trip
down the coast we packed our bags early on saturday morning and hit
the highway before the sun was even up the weather was absolutely
perfect with clear blue skies and a gentle breeze blowing through
the open windows as we drove we played our favorite playlist and
sang along at the top of our lungs around noon we stopped at a
small coastal town to grab some lunch we found a cozy little diner
right by the pier where we ordered fresh fish and chips and homemade
lemonade after eating we walked along the beach collected a few
unique seashells and took tons of pictures to capture the memory it
was honestly one of the best weekends ive had in a long time and i
cannot wait until our next adventure
```

Output text:

```
Last week, my friends and I decided to take a spontaneous road trip
down the coast. We packed our bags early on Saturday morning and hit
the highway before the sun was even up. The weather was absolutely
perfect with clear blue skies and a gentle breeze blowing through
the open windows. As we drove, we played our favorite playlist and
sang along at the top of our lungs. Around noon, we stopped at a
small coastal town to grab some lunch. We found a cozy little diner,
right by the pier, where we ordered fresh fish and chips and homemade
lemonade. After eating, we walked along the beach, collected a few
unique seashells and took tons of pictures to capture the memory. It
was honestly one of the best weekends Ive had in a long time. and I
cannot wait until our next adventure.
```

Performance (web demo, 1 thread): `Words: 145 | Elapsed: 0.085s`

## How to build

### 1. Download the model

```bash
cd flutter-examples/online-punctuation/assets
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
tar xvf sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
rm sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
# We use model.int8.onnx, so delete the larger model.onnx to save space.
rm sherpa-onnx-online-punct-en-2024-08-06/model.onnx
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

Then visit <http://localhost:6006> in your browser.

  - For iOS

Connect your iPhone, then:

```bash
flutter devices
flutter run -d <device-id> --release
```

If you get signing errors, open `ios/Runner.xcworkspace` in Xcode, configure your signing team, then retry.
