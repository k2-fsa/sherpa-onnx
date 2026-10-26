// Copyright 2026 Xiaomi Corporation
//
// This file shows how to use streaming VoxLock speech enhancement models in
// sherpa-onnx.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock passes the audio
// through unchanged until it has heard 3 seconds of speech, takes that speech
// as the speaker to keep, and isolates from then on. See
// StreamingSpeechEnhancementVoxLockWithVoiceSample.java if you have a voice
// sample of the speaker.
//
// Download models and sample test waves from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

import com.k2fsa.sherpa.onnx.*;

public class StreamingSpeechEnhancementVoxLock {
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

    String outFilename = "voxlock-enhanced-online-16k.wav";
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
