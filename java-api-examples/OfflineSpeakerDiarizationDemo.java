// Copyright 2024 Xiaomi Corporation

// This file shows how to use sherpa-onnx Java API for speaker diarization,
import com.k2fsa.sherpa.onnx.*;

public class OfflineSpeakerDiarizationDemo {
  public static void main(String[] args) {
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

    String segmentationModel = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
    String embeddingModel = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
    String waveFilename = "./0-four-speakers-zh.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineSpeakerSegmentationPyannoteModelConfig pyannote =
        OfflineSpeakerSegmentationPyannoteModelConfig.builder().setModel(segmentationModel).build();

    OfflineSpeakerSegmentationModelConfig segmentation =
        OfflineSpeakerSegmentationModelConfig.builder()
            .setPyannote(pyannote)
            .setDebug(true)
            .build();

    SpeakerEmbeddingExtractorConfig embedding =
        SpeakerEmbeddingExtractorConfig.builder().setModel(embeddingModel).setDebug(true).build();

    // The test wave file ./0-four-speakers-zh.wav contains four speakers, so
    // we use numClusters=4 here. If you don't know the number of speakers
    // in the test wave file, please set the numClusters to -1 and provide
    // threshold for clustering
    FastClusteringConfig clustering =
        FastClusteringConfig.builder()
            .setNumClusters(4) // set it to -1 if you don't know the actual number
            .setThreshold(0.5f)
            .build();

    OfflineSpeakerDiarizationConfig config =
        OfflineSpeakerDiarizationConfig.builder()
            .setSegmentation(segmentation)
            .setEmbedding(embedding)
            .setClustering(clustering)
            .setMinDurationOn(0.2f)
            .setMinDurationOff(0.5f)
            .build();

    OfflineSpeakerDiarization sd = new OfflineSpeakerDiarization(config);
    if (sd.getSampleRate() != reader.getSampleRate()) {
      System.out.printf(
          "Expected sample rate: %d, given: %d\n", sd.getSampleRate(), reader.getSampleRate());
      return;
    }

    // OfflineSpeakerDiarizationSegment[] segments = sd.process(reader.getSamples());
    // without callback is also ok

    // or you can use a callback to show the progress
    OfflineSpeakerDiarizationSegment[] segments =
        sd.processWithCallback(
            reader.getSamples(),
            (int numProcessedChunks, int numTotalChunks, long arg) -> {
              float progress = 100.0f * numProcessedChunks / numTotalChunks;
              System.out.printf("Progress: %.2f%%\n", progress);

              return 0;
            });

    for (OfflineSpeakerDiarizationSegment s : segments) {
      System.out.printf("%.3f -- %.3f speaker_%02d\n", s.getStart(), s.getEnd(), s.getSpeaker());
    }

    sd.release();
  }
}
