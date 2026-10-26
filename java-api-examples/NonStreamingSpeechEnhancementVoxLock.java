// Copyright 2026 Xiaomi Corporation

// This file shows how to use VoxLock speech enhancement models in sherpa-onnx.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// NonStreamingSpeechEnhancementVoxLockWithVoiceSample.java if you have a voice
// sample of the speaker.
//
// Download models and sample test waves from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

import com.k2fsa.sherpa.onnx.*;

public class NonStreamingSpeechEnhancementVoxLock {
  public static void main(String[] args) {
    OfflineSpeechDenoiserModelConfig modelConfig =
        OfflineSpeechDenoiserModelConfig.builder()
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .setVoxlock(
                OfflineSpeechDenoiserVoxLockModelConfig.builder()
                    .setEnrollModel("./sherpa-onnx-voxlock-int8/enroll.onnx")
                    .setChunkModel("./sherpa-onnx-voxlock-int8/chunk_int8.onnx")
                    .build())
            .build();
    OfflineSpeechDenoiserConfig config =
        OfflineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

    OfflineSpeechDenoiser speechDenoiser = new OfflineSpeechDenoiser(config);

    WaveReader reader = new WaveReader("./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav");

    long begin = System.currentTimeMillis();
    DenoisedAudio denoised = speechDenoiser.run(reader.getSamples(), reader.getSampleRate());
    long elapsedMs = System.currentTimeMillis() - begin;

    String outFilename = "voxlock-enhanced-16k.wav";
    WaveWriter.write(outFilename, denoised.getSamples(), denoised.getSampleRate());

    float elapsedSeconds = elapsedMs / 1000.0f;
    float duration = reader.getSamples().length / (float) reader.getSampleRate();
    System.out.printf("Saved to %s\n", outFilename);
    System.out.printf("Elapsed seconds: %.3f\n", elapsedSeconds);
    System.out.printf("Audio duration in seconds: %.3f\n", duration);
    System.out.printf(
        "RTF: %.3f/%.3f = %.3f\n", elapsedSeconds, duration, elapsedSeconds / duration);

    speechDenoiser.release();
  }
}
