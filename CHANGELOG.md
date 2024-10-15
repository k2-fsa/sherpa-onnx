## 1.10.28

* Fix swift example for generating subtitles. (#1362)
* Allow more online models to load tokens file from the memory (#1352)
* Fix CI errors introduced by supporting loading keywords from buffers (#1366)
* Fix running MeloTTS models on GPU. (#1379)
* Support Parakeet models from NeMo (#1381)
* Export Pyannote speaker segmentation models to onnx (#1382)
* Support Agglomerative clustering. (#1384)
* Add Python API for clustering (#1385)
* support whisper turbo (#1390)
* context_state is not set correctly when previous context is passed after reset (#1393)
* Speaker diarization example with onnxruntime Python API (#1395)
* C++ API for speaker diarization (#1396)
* Python API for speaker diarization. (#1400)
* C API for speaker diarization (#1402)
* docs(nodejs-addon-examples): add guide for pnpm user (#1401)
* Go API for speaker diarization (#1403)
* Swift API for speaker diarization (#1404)
* Update readme to include more external projects using sherpa-onnx (#1405)
* C# API for speaker diarization (#1407)
* JavaScript API (node-addon) for speaker diarization (#1408)
* WebAssembly exmaple for speaker diarization (#1411)
* Handle audio files less than 10s long for speaker diarization. (#1412)
* JavaScript API with WebAssembly for speaker diarization (#1414)
* Kotlin API for speaker diarization (#1415)
* Java API for speaker diarization (#1416)
* Dart API for speaker diarization (#1418)
* Pascal API for speaker diarization (#1420)
* Android JNI support for speaker diarization (#1421)
* Android demo for speaker diarization (#1423)

## 1.10.27

* Add non-streaming ONNX models for Russian ASR (#1358)
* Fix building Flutter TTS examples for Linux (#1356)
* Support passing utf-8 strings from JavaScript to C++. (#1355)
* Fix sherpa_onnx.go to support returning empty recognition results (#1353)

## 1.10.26

* Add links to projects using sherpa-onnx. (#1345)
* Support lang/emotion/event results from SenseVoice in Swift API. (#1346)
* Support specifying max speech duration for VAD. (#1348)
* Add APIs about max speech duration in VAD for various programming languages (#1349)

## 1.10.25

* Allow tokens and hotwords to be loaded from buffered string driectly (#1339)
* Fix computing features for CED audio tagging models. (#1341)
* Preserve previous result as context for next segment (#1335)
* Add Python binding for online punctuation models (#1312)
* Fix vad.Flush(). (#1329)
* Fix wasm app for streaming paraformer (#1328)
* Build websocket related binaries for embedded systems. (#1327)
* Fixed the C api calls and created the TTS project file (#1324)
* Re-implement LM rescore for online transducer (#1231)

## 1.10.24

* Add VAD and keyword spotting for the Node package with WebAssembly (#1286)
* Fix releasing npm package and fix building Android VAD+ASR example (#1288)
* add Tokens []string, Timestamps []float32, Lang string, Emotion string, Event string (#1277)
* add vad+sense voice example for C API (#1291)
* ADD VAD+ASR example for dart with CircularBuffer. (#1293)
* Fix VAD+ASR example for Dart API. (#1294)
* Avoid SherpaOnnxSpeakerEmbeddingManagerFreeBestMatches freeing null. (#1296)
* Fix releasing wasm app for vad+asr (#1300)
* remove extra files from linux/macos/windows jni libs (#1301)
* two-pass Android APK for SenseVoice (#1302)
* Downgrade flutter sdk versions. (#1305)
* Reduce onnxruntime log output. (#1306)
* Provide prebuilt .jar files for different java versions. (#1307)


## 1.10.23

* flutter: add lang, emotion, event to OfflineRecognizerResult (#1268)
* Use a separate thread to initialize models for lazarus examples. (#1270)
* Object pascal examples for recording and playing audio with portaudio. (#1271)
* Text to speech API for Object Pascal. (#1273)
* update kotlin api for better release native object and add user-friendly apis. (#1275)
* Update wave-reader.cc to support 8/16/32-bit waves (#1278)
* Add WebAssembly for VAD (#1281)
* WebAssembly example for VAD + Non-streaming ASR (#1284)

## 1.10.22

* Add Pascal API for reading wave files (#1243)
* Pascal API for streaming ASR (#1246)
* Pascal API for non-streaming ASR (#1247)
* Pascal API for VAD (#1249)
* Add more C API examples (#1255)
* Add emotion, event of SenseVoice. (#1257)
* Support reading multi-channel wave files with 8/16/32-bit encoded samples (#1258)
* Enable IPO only for Release build. (#1261)
* Add Lazarus example for generating subtitles using Silero VAD with non-streaming ASR (#1251)
* Fix looking up OOVs in lexicon.txt for MeloTTS models. (#1266)


## 1.10.21

* Fix ffmpeg c api example (#1185)
* Fix splitting sentences for MeloTTS (#1186)
* Non-streaming WebSocket client for Java. (#1190)
* Fix copying asset files for flutter examples. (#1191)
* Add Chinese+English tts example for flutter (#1192)
* Add speaker identification and verification exmaple for Dart API (#1194)
* Fix reading non-standard wav files. (#1199)
* Add ReazonSpeech Japanese pre-trained model (#1203)
* Describe how to add new words for MeloTTS models (#1209)
* Remove libonnxruntime_providers_cuda.so as a dependency. (#1210)
* Fix setting SenseVoice language. (#1214)
* Support passing TTS callback in Swift API (#1218)
* Add MeloTTS example for ios (#1223)
* Add online punctuation and casing prediction model for English language (#1224)
* Fix python two pass ASR examples (#1230)
* Add blank penalty for various language bindings

## 1.10.20

* Add Dart API for audio tagging
* Add Dart API for adding punctuations to text

## 1.10.19

* Prefix all C API functions with SherpaOnnx

## 1.10.18

* Fix the case when recognition results contain the symbol `"`. It caused
  issues when converting results to a json string.

## 1.10.17

* Support SenseVoice CTC models.
* Add Dart API for keyword spotter.

## 1.10.16

* Support zh-en TTS model from MeloTTS.

## 1.10.15

* Downgrade onnxruntime from v1.18.1 to v1.17.1

## 1.10.14

* Support whisper large v3
* Update onnxruntime from v1.18.0 to v1.18.1
* Fix invalid utf8 sequence from Whisper for Dart API.

## 1.10.13

* Update onnxruntime from 1.17.1 to 1.18.0
* Add C# API for Keyword spotting

## 1.10.12

* Add Flush to VAD so that the last speech segment can be detected. See also
  https://github.com/k2-fsa/sherpa-onnx/discussions/1077#discussioncomment-9979740

## 1.10.11

* Support the iOS platform for Flutter.

## 1.10.10

* Build sherpa-onnx into a single shared library.

## 1.10.9

* Fix released packages. piper-phonemize was not included in v1.10.8.

## 1.10.8

* Fix released packages. There should be a lib directory.

## 1.10.7

* Support Android for Flutter.

## 1.10.2

* Fix passing C# string to C++

## 1.10.1

* Enable to stop TTS generation

## 1.10.0

* Add inverse text normalization

## 1.9.30

* Add TTS

## 1.9.29

* Publish with CI

## 0.0.3

* Fix path separator on Windows.

## 0.0.2

* Support specifying lib path.

## 0.0.1

* Initial release.
