#!/usr/bin/env bash
#
# Test group: speech enhancement, source separation, speaker
# recognition/diarization, text post-processing, audio tagging.
#
# One of several groups run in parallel by .github/workflows/run-python-test.yaml.
# Run .github/scripts/test-python.sh to run every group sequentially.

set -ex

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

export GIT_CLONE_PROTECTION_ACTIVE=false

log "test version"
python3 ./python-api-examples/version-test.py

log "test offline speech enhancement (VoxLock)"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2

python3 ./python-api-examples/offline-speech-enhancement-voxlock.py
mkdir -p offline-speech-enhancement-voxlock
mv voxlock-enhanced-16k.wav offline-speech-enhancement-voxlock/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav offline-speech-enhancement-voxlock/

python3 ./python-api-examples/offline-speech-enhancement-voxlock-with-voice-sample.py
mkdir -p offline-speech-enhancement-voxlock-with-voice-sample
mv voxlock-enhanced-with-voice-sample-16k.wav offline-speech-enhancement-voxlock-with-voice-sample/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav offline-speech-enhancement-voxlock-with-voice-sample/

python3 ./python-api-examples/online-speech-enhancement-voxlock.py
mkdir -p online-speech-enhancement-voxlock
mv voxlock-enhanced-online-16k.wav online-speech-enhancement-voxlock/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav online-speech-enhancement-voxlock/

python3 ./python-api-examples/online-speech-enhancement-voxlock-with-voice-sample.py
mkdir -p online-speech-enhancement-voxlock-with-voice-sample
mv voxlock-enhanced-online-with-voice-sample-16k.wav online-speech-enhancement-voxlock-with-voice-sample/
cp sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav online-speech-enhancement-voxlock-with-voice-sample/

ls -lh offline-speech-enhancement-voxlock offline-speech-enhancement-voxlock-with-voice-sample online-speech-enhancement-voxlock online-speech-enhancement-voxlock-with-voice-sample

rm -rf sherpa-onnx-voxlock-int8

log "test offline speech enhancement (GTCRN)"

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/speech_with_noise.wav
python3 ./python-api-examples/offline-speech-enhancement-gtcrn.py
python3 ./python-api-examples/offline-speech-enhancement-dpdfnet.py
python3 ./python-api-examples/online-speech-enhancement-gtcrn.py
python3 ./python-api-examples/online-speech-enhancement-dpdfnet.py
ls -lh *.wav

log "test spleeter"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
download https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
./python-api-examples/offline-source-separation-spleeter.py
rm -rf sherpa-onnx-spleeter-2stems-fp16
rm qi-feng-le-zh.wav

log "test UVR"

download https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR_MDXNET_9482.onnx
download https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
./python-api-examples/offline-source-separation-uvr.py
rm UVR_MDXNET_9482.onnx
rm qi-feng-le-zh.wav

mkdir source-separation

mv spleeter-*.wav source-separation
mv uvr-*.wav source-separation

ls -lh source-separation

log "test offline speaker diarization"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

if python3 -c "import librosa" 2>/dev/null; then
  python3 ./python-api-examples/offline-speaker-diarization.py
else
  log "Skipping offline-speaker-diarization.py (librosa not installed)"
fi

rm -rf *.wav *.onnx ./sherpa-onnx-pyannote-segmentation-3-0

log "test_clustering"
pushd /tmp/
mkdir test-cluster
cd test-cluster
download https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
git clone https://github.com/csukuangfj/sr-data
popd

python3 ./sherpa-onnx/python/tests/test_fast_clustering.py

rm -rf /tmp/test-cluster

log "test offline punctuation"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
repo=sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12
ls -lh $repo

python3 ./python-api-examples/add-punctuation.py

rm -rf $repo

log "test offline diacritization"

download_and_extract https://github.com/abjadai/catt/releases/download/v2/eo_model_onnx.zip catt_eo_model_onnx

python3 ./python-api-examples/add-diacritics.py

rm -rf catt_eo_model_onnx

log "test online punctuation"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
repo=sherpa-onnx-online-punct-en-2024-08-06
ls -lh $repo

python3 ./python-api-examples/add-punctuation-online.py

rm -rf $repo

log "test audio tagging"

download_and_extract https://github.com/k2-fsa/sherpa-onnx/releases/download/audio-tagging-models/sherpa-onnx-zipformer-audio-tagging-2024-04-09.tar.bz2
 python3 ./python-api-examples/audio-tagging-from-a-file.py
rm -rf sherpa-onnx-zipformer-audio-tagging-2024-04-09

log "test speaker recognition (python)"
.github/scripts/test-speaker-recognition-python.sh

log "test-python-misc.sh passed"
