// Copyright 2025 Xiaomi Corporation

// This file shows how to use speech enhancement models in sherpa-onnx
//
// please download files in this script from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

import com.k2fsa.sherpa.onnx.*;

public class NonStreamingSpeechEnhancementGtcrn {
  public static void main(String[] args) {
    String model = "./gtcrn_simple.onnx";
    OfflineSpeechDenoiserGtcrnModelConfig gtcrn =
        OfflineSpeechDenoiserGtcrnModelConfig.builder().setModel(model).build();

    OfflineSpeechDenoiserModelConfig modelConfig =
        OfflineSpeechDenoiserModelConfig.builder()
            .setGtcrn(gtcrn)
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .build();
    OfflineSpeechDenoiserConfig config =
        OfflineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

    OfflineSpeechDenoiser speech_denoiser = new OfflineSpeechDenoiser(config);

    String testWaveFilename = "./inp_16k.wav";
    WaveReader reader = new WaveReader(testWaveFilename);

    DenoisedAudio denoised = speech_denoiser.run(reader.getSamples(), reader.getSampleRate());
    String outFilename = "enhanced-16k.wav";
    WaveWriter.write(outFilename, denoised.getSamples(), denoised.getSampleRate());
    System.out.printf("Saved to %s\n", outFilename);

    speech_denoiser.release();
  }
}
