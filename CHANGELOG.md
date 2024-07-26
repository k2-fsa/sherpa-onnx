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
