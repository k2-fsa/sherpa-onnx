# sherpa-onnx app example

## Pre-built Flutter apps

Download pre-built Flutter apps for different platforms at
https://github.com/k2-fsa/sherpa-onnx/releases/tag/flutter

### VAD from file

| Platform | File |
|---|---|
| Android (arm64-v8a) | `flutter-vad-from-file-arm64-v8a.apk` |
| Android (armeabi-v7a) | `flutter-vad-from-file-armeabi-v7a.apk` |
| Android (x86_64) | `flutter-vad-from-file-x86_64.apk` |
| Linux (x64) | `flutter-vad-from-file-linux-x64.zip` |
| Windows (x64) | `flutter-vad-from-file-win-x64.zip` |
| Web | `flutter-vad-from-file-web.zip` |

### VAD from microphone

| Platform | File |
|---|---|
| Android (arm64-v8a) | `flutter-vad-from-microphone-arm64-v8a.apk` |
| Android (armeabi-v7a) | `flutter-vad-from-microphone-armeabi-v7a.apk` |
| Android (x86_64) | `flutter-vad-from-microphone-x86_64.apk` |
| Linux (x64) | `flutter-vad-from-microphone-linux-x64.zip` |
| macOS | `flutter-vad-from-microphone-macos.zip` |
| Windows (x64) | `flutter-vad-from-microphone-win-x64.zip` |
| Web | `flutter-vad-from-microphone-web.zip` |

### Offline punctuation restoration

| Platform | File |
|---|---|
| Android (arm64-v8a) | `flutter-offline-punctuation-arm64-v8a.apk` |
| Android (armeabi-v7a) | `flutter-offline-punctuation-armeabi-v7a.apk` |
| Android (x86_64) | `flutter-offline-punctuation-x86_64.apk` |
| Linux (x64) | `flutter-offline-punctuation-linux-x64.zip` |
| macOS | `flutter-offline-punctuation-macos.zip` |
| Windows (x64) | `flutter-offline-punctuation-win-x64.zip` |
| Web | `flutter-offline-punctuation-web.zip` |

### Online punctuation restoration

| Platform | File |
|---|---|
| Android (arm64-v8a) | `flutter-online-punctuation-arm64-v8a.apk` |
| Android (armeabi-v7a) | `flutter-online-punctuation-armeabi-v7a.apk` |
| Android (x86_64) | `flutter-online-punctuation-x86_64.apk` |
| Linux (x64) | `flutter-online-punctuation-linux-x64.zip` |
| macOS | `flutter-online-punctuation-macos.zip` |
| Windows (x64) | `flutter-online-punctuation-win-x64.zip` |
| Web | `flutter-online-punctuation-web.zip` |

## Web demos

Try the following demos directly in your browser:

| Demo | URL |
|---|---|
| VAD from file | https://modelscope.cn/studios/csukuangfj/wasm-vad-from-file |
| VAD from microphone | https://modelscope.cn/studios/csukuangfj/wasm-vad-from-microphone |
| Offline punctuation restoration | https://modelscope.cn/studios/csukuangfj/wasm-offline-punctuation |
| Online punctuation restoration | https://modelscope.cn/studios/csukuangfj/wasm-online-punctuation |

## Flutter source code

| Functions | URL | Supported Platforms|
|---|---|---|
|Hello world (version info)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/hello_world)| Android, iOS, Linux, macOS, Windows, **Web**|
|Streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/streaming_asr)| Android, iOS, Linux, macOS, Windows|
|Non-streaming VAD + speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/non_streaming_vad_asr)| Android, iOS, Linux, macOS, Windows|
|Text to speech| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/tts)| Android, iOS, Linux, macOS, Windows, **Web**|
|VAD from file| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/vad-from-file)| Android, iOS, Linux, macOS, Windows, **Web**|
|VAD from microphone| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/vad-from-microphone)| Android, iOS, Linux, macOS, Windows, **Web**|
|Offline punctuation restoration| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/offline-punctuation)| Android, iOS, Linux, macOS, Windows, **Web**|
|Online punctuation restoration| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/flutter-examples/online-punctuation)| Android, iOS, Linux, macOS, Windows, **Web**|

## Pure dart-examples

Hint: All of the following functions can be used in Flutter, even if some of them are only provided in pure dart api examples.

| Functions | URL | Supported Platforms|
|---|---|---|
|Streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/streaming-asr)| macOS, Windows, Linux|
|Non-Streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/non-streaming-asr)| macOS, Windows, Linux|
|Text to speech| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/tts)| macOS, Windows, Linux|
|Voice activity detection (VAD)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad)| macOS, Windows, Linux|
|VAD with non-streaming speech recognition| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/vad-with-non-streaming-asr)| macOS, Windows, Linux|
|Speaker identification and verification| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speaker-identification)| macOS, Windows, Linux|
|Speaker diarization| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speaker-diarization)| macOS, Windows, Linux|
|Audio tagging| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/audio-tagging)| macOS, Windows, Linux|
|Keyword spotter| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/keyword-spotter)| macOS, Windows, Linux|
|Spoken language identification| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/spoken-language-identification)| macOS, Windows, Linux|
|Add punctuations| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/add-punctuations)| macOS, Windows, Linux|
|Speech enhancement (GTCRN)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speech-enhancement-gtcrn)| macOS, Windows, Linux|
|Speech enhancement (DPDFNet)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/speech-enhancement-dpdfnet)| macOS, Windows, Linux|
|Streaming speech enhancement (GTCRN)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/streaming-speech-enhancement-gtcrn)| macOS, Windows, Linux|
|Streaming speech enhancement (DPDFNet)| [Address](https://github.com/k2-fsa/sherpa-onnx/tree/master/dart-api-examples/streaming-speech-enhancement-dpdfnet)| macOS, Windows, Linux|
