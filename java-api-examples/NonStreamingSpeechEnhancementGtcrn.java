// Copyright 2025 Xiaomi Corporation

// This file shows how to use speech enhancement models in sherpa-onnx
//
// Download GTCRN models and sample test waves from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

import com.k2fsa.sherpa.onnx.*;

public class NonStreamingSpeechEnhancementGtcrn {
  public static void main(String[] args) {
    String model = "./gtcrn_simple.onnx";
    OfflineSpeechDenoiserModelConfig.Builder builder =
        OfflineSpeechDenoiserModelConfig.builder()
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu");

    builder.setGtcrn(OfflineSpeechDenoiserGtcrnModelConfig.builder().setModel(model).build());

    OfflineSpeechDenoiserModelConfig modelConfig = builder.build();
    OfflineSpeechDenoiserConfig config =
        OfflineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

    OfflineSpeechDenoiser speechDenoiser = new OfflineSpeechDenoiser(config);

    String testWaveFilename = "./inp_16k.wav";
    WaveReader reader = new WaveReader(testWaveFilename);

    DenoisedAudio denoised = speechDenoiser.run(reader.getSamples(), reader.getSampleRate());
    String outFilename = "enhanced.wav";
    WaveWriter.write(outFilename, denoised.getSamples(), denoised.getSampleRate());
    System.out.printf("Saved to %s\n", outFilename);

    speechDenoiser.release();
  }
}
