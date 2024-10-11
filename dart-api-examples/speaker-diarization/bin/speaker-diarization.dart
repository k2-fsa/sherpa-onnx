// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';
import 'dart:ffi';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import './init.dart';

void main(List<String> arguments) async {
  await initSherpaOnnx();

  /* Please use the following commands to download files used in this file
    Step 1: Download a speaker segmentation model

    Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
    for a list of available models. The following is an example

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
      tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
      rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

    Step 2: Download a speaker embedding extractor model

    Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
    for a list of available models. The following is an example

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

    Step 3. Download test wave files

    Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
    for a list of available test wave files. The following is an example

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

    Step 4. Run it
        */

  final segmentationModel =
      "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx";

  final embeddingModel =
      "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

  final waveFilename = "./0-four-speakers-zh.wav";

  final segmentationConfig = sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
    pyannote: sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
        model: segmentationModel),
  );

  final embeddingConfig =
      sherpa_onnx.SpeakerEmbeddingExtractorConfig(model: embeddingModel);

  // since we know there are 4 speakers in ./0-four-speakers-zh.wav, we set
  // numClusters to 4. If you don't know the exact number, please set it to -1.
  // in that case, you have to set threshold. A larger threshold leads to
  // fewer clusters, i.e., fewer speakers.
  final clusteringConfig =
      sherpa_onnx.FastClusteringConfig(numClusters: 4, threshold: 0.5);

  var config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
      segmentation: segmentationConfig,
      embedding: embeddingConfig,
      clustering: clusteringConfig,
      minDurationOn: 0.2,
      minDurationOff: 0.5);

  final sd = sherpa_onnx.OfflineSpeakerDiarization(config);
  if (sd.ptr == nullptr) {
    return;
  }

  final waveData = sherpa_onnx.readWave(waveFilename);
  if (sd.sampleRate != waveData.sampleRate) {
    print(
        'Expected sample rate: ${sd.sampleRate}, given: ${waveData.sampleRate}');
    return;
  }

  print('started');

  // Use the following statement if you don't want to use a callback
  // final segments = sd.process(samples: waveData.samples);

  final segments = sd.processWithCallback(
      samples: waveData.samples,
      callback: (int numProcessedChunk, int numTotalChunks) {
        final progress = 100.0 * numProcessedChunk / numTotalChunks;

        print('Progress ${progress.toStringAsFixed(2)}%');

        return 0;
      });

  for (int i = 0; i < segments.length; ++i) {
    print(
        '${segments[i].start.toStringAsFixed(3)} -- ${segments[i].end.toStringAsFixed(3)}  speaker_${segments[i].speaker}');
  }
}
