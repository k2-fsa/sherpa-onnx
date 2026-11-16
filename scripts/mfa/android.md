# Forced-alignment Android app (APKs)

This page describes the APK files of the
[apk-fa](https://github.com/k2-fsa/sherpa-onnx/releases/tag/apk-fa) GitHub
release. Each APK is a build of the app in
[`android/SherpaOnnxForcedAlignment`](../../android/SherpaOnnxForcedAlignment)
with one forced-alignment model bundled in.

## Table of contents

- [What the app does](#what-the-app-does)
- [Which APK do I need?](#which-apk-do-i-need)
- [How to build it yourself](#how-to-build-it-yourself)

## What the app does

- **Pick an audio (or video) file** — wav, mp3, m4a/aac, flac, ogg, opus, wma,
  aiff, mp4, mov, webm, mkv, ... (anything FFmpeg can decode). Or tap
  **Example** to load the model's bundled test wav and its transcript, or
  **Record** from the microphone, so the app can be tried without any external
  data.
- **Enter the transcript** of the audio. Chinese text is tokenized per hanzi,
  English words are split on whitespace; punctuation is ignored. Words that
  are not in the model's lexicon are reported as out-of-vocabulary.
- **Align** — the app runs the forced aligner on-device and shows a table of
  the words with their start/end times, plus time statistics (audio duration,
  decode / align time, RTF). Click a row to play just that word — cut
  sample-accurately from the audio, so it never includes the neighbors. The
  result can be exported as a Praat **TextGrid**, a timestamped **.txt** or a
  plain **.txt**, or shared with other apps (WeChat, email, ...) as
  `.TextGrid.txt` / `.words.txt`.
- The **Info** tab shows the version of sherpa-onnx / onnxruntime and links to
  the model and its documentation.

Everything runs on the device; no audio is uploaded anywhere.

## Which APK do I need?

Every APK is named `sherpa-onnx-fa-<abi>-<model>.apk`. Pick your device's ABI
first, then the model by language:

- **`arm64-v8a`** — most modern phones and tablets. **If you are unsure,
  pick this one.**
- **`armeabi-v7a`** — older 32-bit ARM phones.
- **`x86_64`**, **`x86`** — x86-based devices.

If the ABI does not match your device, Android refuses to install the APK.

The table below is for `arm64-v8a`; the collapsed sections below it list the
same models for the other ABIs.

| APK | Language | Model | Notes |
|-----|----------|-------|-------|
| `sherpa-onnx-fa-arm64-v8a-zh-aishell3.apk` | Chinese | [zh-aishell3-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 |
| `sherpa-onnx-fa-arm64-v8a-zh-biaobei.apk` | Chinese | [zh-biaobei-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-biaobei-pinyin.tar.bz2) | pinyin lexicon, trained on the biaobei reading corpus |
| `sherpa-onnx-fa-arm64-v8a-zh-large.apk` | Chinese | [zh-large-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-large-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 + biaobei + more |
| `sherpa-onnx-fa-arm64-v8a-en-v2.2.1.apk` | English | [english-mfa-v2.2.1](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v2.2.1.tar.bz2) | MFA `english_mfa` acoustic model v2.2.1, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-arm64-v8a-en-v3.0.0.apk` | English | [english-mfa-v3.0.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.0.0.tar.bz2) | MFA `english_mfa` acoustic model v3.0.0, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-arm64-v8a-en-v3.1.0.apk` | English | [english-mfa-v3.1.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2) | MFA `english_mfa` acoustic model v3.1.0, 100 phones, 42k-word core dictionary |

<details>
<summary>APK names for <code>armeabi-v7a</code> (older 32-bit ARM phones) — click to show</summary>

| APK | Language | Model | Notes |
|-----|----------|-------|-------|
| `sherpa-onnx-fa-armeabi-v7a-zh-aishell3.apk` | Chinese | [zh-aishell3-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 |
| `sherpa-onnx-fa-armeabi-v7a-zh-biaobei.apk` | Chinese | [zh-biaobei-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-biaobei-pinyin.tar.bz2) | pinyin lexicon, trained on the biaobei reading corpus |
| `sherpa-onnx-fa-armeabi-v7a-zh-large.apk` | Chinese | [zh-large-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-large-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 + biaobei + more |
| `sherpa-onnx-fa-armeabi-v7a-en-v2.2.1.apk` | English | [english-mfa-v2.2.1](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v2.2.1.tar.bz2) | MFA `english_mfa` acoustic model v2.2.1, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-armeabi-v7a-en-v3.0.0.apk` | English | [english-mfa-v3.0.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.0.0.tar.bz2) | MFA `english_mfa` acoustic model v3.0.0, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-armeabi-v7a-en-v3.1.0.apk` | English | [english-mfa-v3.1.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2) | MFA `english_mfa` acoustic model v3.1.0, 100 phones, 42k-word core dictionary |

</details>

<details>
<summary>APK names for <code>x86_64</code> — click to show</summary>

| APK | Language | Model | Notes |
|-----|----------|-------|-------|
| `sherpa-onnx-fa-x86_64-zh-aishell3.apk` | Chinese | [zh-aishell3-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 |
| `sherpa-onnx-fa-x86_64-zh-biaobei.apk` | Chinese | [zh-biaobei-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-biaobei-pinyin.tar.bz2) | pinyin lexicon, trained on the biaobei reading corpus |
| `sherpa-onnx-fa-x86_64-zh-large.apk` | Chinese | [zh-large-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-large-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 + biaobei + more |
| `sherpa-onnx-fa-x86_64-en-v2.2.1.apk` | English | [english-mfa-v2.2.1](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v2.2.1.tar.bz2) | MFA `english_mfa` acoustic model v2.2.1, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-x86_64-en-v3.0.0.apk` | English | [english-mfa-v3.0.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.0.0.tar.bz2) | MFA `english_mfa` acoustic model v3.0.0, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-x86_64-en-v3.1.0.apk` | English | [english-mfa-v3.1.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2) | MFA `english_mfa` acoustic model v3.1.0, 100 phones, 42k-word core dictionary |

</details>

<details>
<summary>APK names for <code>x86</code> — click to show</summary>

| APK | Language | Model | Notes |
|-----|----------|-------|-------|
| `sherpa-onnx-fa-x86-zh-aishell3.apk` | Chinese | [zh-aishell3-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 |
| `sherpa-onnx-fa-x86-zh-biaobei.apk` | Chinese | [zh-biaobei-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-biaobei-pinyin.tar.bz2) | pinyin lexicon, trained on the biaobei reading corpus |
| `sherpa-onnx-fa-x86-zh-large.apk` | Chinese | [zh-large-pinyin](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-large-pinyin.tar.bz2) | pinyin lexicon, trained on AISHELL-3 + biaobei + more |
| `sherpa-onnx-fa-x86-en-v2.2.1.apk` | English | [english-mfa-v2.2.1](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v2.2.1.tar.bz2) | MFA `english_mfa` acoustic model v2.2.1, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-x86-en-v3.0.0.apk` | English | [english-mfa-v3.0.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.0.0.tar.bz2) | MFA `english_mfa` acoustic model v3.0.0, 100 phones, ~95k-word dictionary |
| `sherpa-onnx-fa-x86-en-v3.1.0.apk` | English | [english-mfa-v3.1.0](https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2) | MFA `english_mfa` acoustic model v3.1.0, 100 phones, 42k-word core dictionary |

</details>

Each model package is built by this directory's scripts; see
[README.md](README.md) for what is inside a package and where the models come
from. The Chinese models are also interchangeable with each other; in general
`zh-large` is the most robust, `zh-aishell3`/`zh-biaobei` match their training
domain. For English, prefer `en-v3.1.0` unless you need the larger dictionary
of v2.2.1/v3.0.0 (see the README for the differences).

## How to build it yourself

1. Build `libsherpa-onnx-jni.so` for your ABI, e.g.:

   ```bash
   ./build-android-arm64-v8a.sh
   ```

2. Download and extract a model package into the app's assets:

   ```bash
   cd android/SherpaOnnxForcedAlignment/app/src/main/assets
   curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/english-mfa-v3.1.0.tar.bz2
   tar xvf english-mfa-v3.1.0.tar.bz2
   cd ../../../../../../  # back to the repository root
   ```

3. Set `SELECTED_MODEL_INDEX` in
   `android/SherpaOnnxForcedAlignment/app/src/main/java/com/k2fsa/sherpa/onnx/fa/ModelConfig.kt`
   to the index of that model (the order in `modelInfo`).

4. Copy the `.so` and build:

   ```bash
   cp build-android-arm64-v8a/install/lib/*.so \
      android/SherpaOnnxForcedAlignment/app/src/main/jniLibs/arm64-v8a/
   cd android/SherpaOnnxForcedAlignment
   ./gradlew build -Pabi=arm64-v8a
   ```

The APK is in `app/build/outputs/apk/debug/`. The CI does exactly this via
[`scripts/apk/build-apk-forced-alignment.sh.in`](../apk/build-apk-forced-alignment.sh.in)
and [`generate-forced-alignment-apk-script.py`](../apk/generate-forced-alignment-apk-script.py);
the workflow is [`.github/workflows/apk-fa.yaml`](../../.github/workflows/apk-fa.yaml).
