// Copyright (c)  2024  Xiaomi Corporation
const sherpa_onnx = require('sherpa-onnx');

// clang-format off
/* Please use the following commands to download files
   used in this script

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

 */
// clang-format on

const config = {
  segmentation: {
    pyannote: {
      model: './sherpa-onnx-pyannote-segmentation-3-0/model.onnx',
      debug: 1,
    },
  },
  embedding: {
    model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
    debug: 1,
  },
  clustering: {
    // since we know that the test wave file
    // ./0-four-speakers-zh.wav contains 4 speakers, we use 4 for numClusters
    // here. if you don't have such information, please set numClusters to -1
    numClusters: 4,

    // If numClusters is not -1, then threshold is ignored.
    //
    // A larger threshold leads to fewer clusters, i.e., fewer speakers
    // A smaller threshold leads to more clusters, i.e., more speakers
    // You need to tune it by yourself.
    threshold: 0.5,
  },

  // If a segment is shorter than minDurationOn, we discard it
  minDurationOn: 0.2,  // in seconds

  // If the gap between two segments is less than minDurationOff, then we
  // merge these two segments into a single one
  minDurationOff: 0.5,  // in seconds
};

const waveFilename = './0-four-speakers-zh.wav';

const sd = sherpa_onnx.createOfflineSpeakerDiarization(config);
console.log('Started')

const wave = sherpa_onnx.readWave(waveFilename);
if (sd.sampleRate != wave.sampleRate) {
  throw new Error(
      `Expected sample rate: ${sd.sampleRate}, given: ${wave.sampleRate}`);
}

const segments = sd.process(wave.samples);
console.log(segments);
