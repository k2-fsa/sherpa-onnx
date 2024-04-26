#!/usr/bin/env bash
#
# This scripts shows how to build JNI libs for sherpa-onnx
# Note: This scripts runs only on Linux and macOS, though sherpa-onnx
# supports building JNI libs for Windows.

set -ex

if [[ ! -f ../build/lib/libsherpa-onnx-jni.dylib  && ! -f ../build/lib/libsherpa-onnx-jni.so ]]; then
  mkdir -p ../build
  pushd ../build
  cmake \
    -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
    -DSHERPA_ONNX_ENABLE_TESTS=OFF \
    -DSHERPA_ONNX_ENABLE_CHECK=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
    -DSHERPA_ONNX_ENABLE_JNI=ON \
    ..

  make -j4
  ls -lh lib
  popd
fi

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

function testSpeakerEmbeddingExtractor() {
  if [ ! -f ./3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_large_sv_zh-cn_3dspeaker_16k.onnx
  fi

  if [ ! -f ./speaker1_a_cn_16k.wav ]; then
    curl -SL -O https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_a_cn_16k.wav
  fi

  if [ ! -f ./speaker1_b_cn_16k.wav ]; then
    curl -SL -O https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker1_b_cn_16k.wav
  fi

  if [ ! -f ./speaker2_a_cn_16k.wav ]; then
    curl -SL -O https://github.com/csukuangfj/sr-data/raw/main/test/3d-speaker/speaker2_a_cn_16k.wav
  fi

  out_filename=test_speaker_id.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_speaker_id.kt \
    OnlineStream.kt \
    Speaker.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}


function testOnlineAsr() {
  if [ ! -f ./sherpa-onnx-streaming-zipformer-en-2023-02-21/tokens.txt ]; then
    git lfs install
    git clone https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21
  fi

  if [ ! -d ./sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13 ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
    rm sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
  fi

  if [ ! -d ./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18 ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
    tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
    rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
  fi

  out_filename=test_online_asr.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_online_asr.kt \
    FeatureConfig.kt \
    OnlineRecognizer.kt \
    OnlineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

function testTts() {
  if [ ! -f ./vits-piper-en_US-amy-low/en_US-amy-low.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
    tar xf vits-piper-en_US-amy-low.tar.bz2
    rm vits-piper-en_US-amy-low.tar.bz2
  fi

  out_filename=test_tts.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_tts.kt \
    Tts.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}


function testAudioTagging() {
  if [ ! -d sherpa-onnx-zipformer-audio-tagging-2024-04-09 ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
    tar xvf sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
    rm sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
  fi

  out_filename=test_audio_tagging.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_audio_tagging.kt \
    AudioTagging.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}


function testSpokenLanguageIdentification() {
  if [ ! -f ./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
    tar xvf sherpa-onnx-whisper-tiny.tar.bz2
    rm sherpa-onnx-whisper-tiny.tar.bz2
  fi

  if [ ! -f ./spoken-language-identification-test-wavs/ar-arabic.wav ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2
    tar xvf spoken-language-identification-test-wavs.tar.bz2
    rm spoken-language-identification-test-wavs.tar.bz2
  fi

  out_filename=test_language_id.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_language_id.kt \
    SpokenLanguageIdentification.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

function testOfflineAsr() {
  if [ ! -f ./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.en.tar.bz2
    tar xvf sherpa-onnx-whisper-tiny.en.tar.bz2
    rm sherpa-onnx-whisper-tiny.en.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-nemo-ctc-en-citrinet-512/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
    tar xvf sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
    rm sherpa-onnx-nemo-ctc-en-citrinet-512.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
    tar xvf sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
    rm sherpa-onnx-paraformer-zh-2023-03-28.tar.bz2
  fi

  if [ ! -f ./sherpa-onnx-zipformer-multi-zh-hans-2023-9-2/tokens.txt ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
    tar xvf sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
    rm sherpa-onnx-zipformer-multi-zh-hans-2023-9-2.tar.bz2
  fi

  out_filename=test_offline_asr.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    test_offline_asr.kt \
    FeatureConfig.kt \
    OfflineRecognizer.kt \
    OfflineStream.kt \
    WaveReader.kt \
    faked-asset-manager.kt

  ls -lh $out_filename
  java -Djava.library.path=../build/lib -jar $out_filename
}

function testPunctuation() {
  if [ ! -f ./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx ]; then
    curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
    tar xvf sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
    rm sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
  fi

  out_filename=test_punctuation.jar
  kotlinc-jvm -include-runtime -d $out_filename \
    ./test_punctuation.kt \
    ./OfflinePunctuation.kt \
    faked-asset-manager.kt \
    faked-log.kt

  ls -lh $out_filename

  java -Djava.library.path=../build/lib -jar $out_filename
}

testSpeakerEmbeddingExtractor
testOnlineAsr
testTts
testAudioTagging
testSpokenLanguageIdentification
testOfflineAsr
testPunctuation
