#!/usr/bin/env bash

set -ex

cd dart-api-examples

pushd speaker-identification
echo '----------3d speaker----------'
./run-3d-speaker.sh
popd

pushd add-punctuations
echo '----------CT Transformer----------'
./run-ct-transformer.sh
popd

pushd audio-tagging
echo '----------zipformer----------'
./run-zipformer.sh

echo '----------ced----------'
./run-ced.sh
popd

pushd vad-with-non-streaming-asr
echo '----------TeleSpeech CTC----------'
./run-telespeech-ctc.sh
rm -rf sherpa-onnx-*

echo "----zipformer transducer----"
./run-zipformer-transducer.sh
rm -rf sherpa-onnx-*

echo "----whisper----"
./run-whisper.sh
rm -rf sherpa-onnx-*

echo "----paraformer----"
./run-paraformer.sh
rm -rf sherpa-onnx-*

echo "----SenseVoice zh----"
./run-sense-voice-zh-2.sh
./run-sense-voice-zh.sh
rm -rf sherpa-onnx-*

echo "----SenseVoice en----"
./run-sense-voice-en.sh
rm -rf sherpa-onnx-*

popd

pushd keyword-spotter
./run-zh.sh
popd

pushd non-streaming-asr

echo '----------SenseVoice----------'
./run-sense-voice.sh
rm -rf sherpa-onnx-*

echo '----------NeMo transducer----------'
./run-nemo-transducer.sh
rm -rf sherpa-onnx-*

echo '----------NeMo CTC----------'
./run-nemo-ctc.sh
rm -rf sherpa-onnx-*

echo '----------TeleSpeech CTC----------'
./run-telespeech-ctc.sh
rm -rf sherpa-onnx-*

echo '----------whisper----------'
./run-whisper.sh
rm -rf sherpa-onnx-*

echo '----------zipformer transducer----------'
./run-zipformer-transducer.sh
rm -rf sherpa-onnx-*

echo '----------paraformer itn----------'
./run-paraformer-itn.sh

echo '----------paraformer----------'
./run-paraformer.sh
rm -rf sherpa-onnx-*

echo '----------VAD with paraformer----------'
./run-vad-with-paraformer.sh
rm -rf sherpa-onnx-*

popd # non-streaming-asr

pushd tts

echo '----------piper tts----------'
./run-piper.sh
rm -rf vits-piper-*

echo '----------coqui tts----------'
./run-coqui.sh
rm -rf vits-coqui-*

echo '----------zh tts----------'
./run-zh.sh
rm -rf sherpa-onnx-*

popd # tts

pushd streaming-asr

echo '----------streaming zipformer ctc HLG----------'
./run-zipformer-ctc-hlg.sh
rm -rf sherpa-onnx-*

echo '----------streaming zipformer ctc----------'
./run-zipformer-ctc.sh
rm -rf sherpa-onnx-*

echo '----------streaming zipformer transducer----------'
./run-zipformer-transducer-itn.sh
./run-zipformer-transducer.sh
rm -f itn*
rm -rf sherpa-onnx-*

echo '----------streaming NeMo transducer----------'
./run-nemo-transducer.sh
rm -rf sherpa-onnx-*

echo '----------streaming paraformer----------'
./run-paraformer.sh
rm -rf sherpa-onnx-*

popd # streaming-asr

pushd vad
./run.sh
rm *.onnx
popd

