// Copyright 2026 Xiaomi Corporation
//
// This file shows how to use streaming VoxLock speech enhancement models in
// sherpa-onnx with a voice sample.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example provides a voice sample of the speaker to keep, so VoxLock
// isolates that speaker from the first sample. The voice sample must have at
// least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
// loudest 3 seconds within its first 30 seconds. See
// StreamingSpeechEnhancementVoxLock.java if you don't have a voice sample.
//
// Download models and sample test waves from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

import com.k2fsa.sherpa.onnx.*;

public class StreamingSpeechEnhancementVoxLockWithVoiceSample {
  private static void appendSamples(java.util.ArrayList<Float> dst, float[] src) {
    for (float v : src) {
      dst.add(v);
    }
  }

  private static float[] toFloatArray(java.util.ArrayList<Float> src) {
    float[] ans = new float[src.size()];
    for (int i = 0; i != src.size(); ++i) {
      ans[i] = src.get(i);
    }
    return ans;
  }

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
                    .setVoiceSample("./sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav")
                    .build())
            .build();

    OnlineSpeechDenoiserConfig config =
        OnlineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

    OnlineSpeechDenoiser speechDenoiser = new OnlineSpeechDenoiser(config);

    WaveReader reader = new WaveReader("./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav");
    int frameShift = speechDenoiser.getFrameShiftInSamples();
    java.util.ArrayList<Float> output = new java.util.ArrayList<>();

    float[] samples = reader.getSamples();
    long begin = System.currentTimeMillis();
    for (int start = 0; start < samples.length; start += frameShift) {
      int end = Math.min(start + frameShift, samples.length);
      float[] chunk = java.util.Arrays.copyOfRange(samples, start, end);
      DenoisedAudio denoised = speechDenoiser.run(chunk, reader.getSampleRate());
      appendSamples(output, denoised.getSamples());
    }

    DenoisedAudio denoised = speechDenoiser.flush();
    appendSamples(output, denoised.getSamples());
    long elapsedMs = System.currentTimeMillis() - begin;

    String outFilename = "voxlock-enhanced-online-with-voice-sample-16k.wav";
    WaveWriter.write(outFilename, toFloatArray(output), speechDenoiser.getSampleRate());

    float elapsedSeconds = elapsedMs / 1000.0f;
    float duration = samples.length / (float) reader.getSampleRate();
    System.out.printf("Saved to %s\n", outFilename);
    System.out.printf("Elapsed seconds: %.3f\n", elapsedSeconds);
    System.out.printf("Audio duration in seconds: %.3f\n", duration);
    System.out.printf(
        "RTF: %.3f/%.3f = %.3f\n", elapsedSeconds, duration, elapsedSeconds / duration);

    speechDenoiser.release();
  }
}
