// Copyright 2026 Silvio Tomatis

// This file shows how to use sherpa-onnx Java API for speaker diarization
// with Nemotron-3-Diarization, an end-to-end streaming Sortformer model.
// It needs neither a speaker embedding model nor clustering.
import com.k2fsa.sherpa.onnx.*;

public class OfflineSpeakerDiarizationSortformerDemo {
  public static void main(String[] args) {
    /* Please use the following commands to download files used in this file
    Step 1: Download the model

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-nemotron-3-diarization.tar.bz2
      tar xvf sherpa-onnx-nemotron-3-diarization.tar.bz2
      rm sherpa-onnx-nemotron-3-diarization.tar.bz2

    Step 2. Download test wave files

      wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

    Step 3. Run it
        */

    String model = "./sherpa-onnx-nemotron-3-diarization/model.int8.onnx";
    String waveFilename = "./0-four-speakers-zh.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineSpeakerSegmentationSortformerModelConfig sortformer =
        OfflineSpeakerSegmentationSortformerModelConfig.builder()
            .setModel(model)
            .setThreshold(0.5f)
            .build();

    OfflineSpeakerSegmentationModelConfig segmentation =
        OfflineSpeakerSegmentationModelConfig.builder()
            .setSortformer(sortformer)
            .setNumThreads(2)
            .setDebug(false)
            .build();

    OfflineSpeakerDiarizationConfig config =
        OfflineSpeakerDiarizationConfig.builder()
            .setSegmentation(segmentation)
            .setMinDurationOn(0.3f)
            .setMinDurationOff(0.5f)
            .build();

    OfflineSpeakerDiarization sd = new OfflineSpeakerDiarization(config);
    if (sd.getSampleRate() != reader.getSampleRate()) {
      System.out.printf(
          "Expected sample rate: %d, given: %d\n", sd.getSampleRate(), reader.getSampleRate());
      sd.release();
      return;
    }

    OfflineSpeakerDiarizationSegment[] segments = sd.process(reader.getSamples());

    for (OfflineSpeakerDiarizationSegment s : segments) {
      System.out.printf("%.3f -- %.3f speaker_%02d\n", s.getStart(), s.getEnd(), s.getSpeaker());
    }

    sd.release();
  }
}
