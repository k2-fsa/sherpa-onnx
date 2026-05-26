// Copyright 2026 Xiaomi Corporation
//
// This file shows how to use streaming GTCRN speech enhancement models in
// sherpa-onnx.
//
// Download GTCRN models and sample test waves from:
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models

import com.k2fsa.sherpa.onnx.*;

public class StreamingSpeechEnhancementGtcrn {
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
    String model = "./gtcrn_simple.onnx";

    OfflineSpeechDenoiserModelConfig modelConfig =
        OfflineSpeechDenoiserModelConfig.builder()
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .setGtcrn(
                OfflineSpeechDenoiserGtcrnModelConfig.builder().setModel(model).build())
            .build();

    OnlineSpeechDenoiserConfig config =
        OnlineSpeechDenoiserConfig.builder().setModel(modelConfig).build();

    OnlineSpeechDenoiser speechDenoiser = new OnlineSpeechDenoiser(config);

    WaveReader reader = new WaveReader("./inp_16k.wav");
    int frameShift = speechDenoiser.getFrameShiftInSamples();
    java.util.ArrayList<Float> output = new java.util.ArrayList<>();

    float[] samples = reader.getSamples();
    for (int start = 0; start < samples.length; start += frameShift) {
      int end = Math.min(start + frameShift, samples.length);
      float[] chunk = java.util.Arrays.copyOfRange(samples, start, end);
      DenoisedAudio denoised = speechDenoiser.run(chunk, reader.getSampleRate());
      appendSamples(output, denoised.getSamples());
    }

    DenoisedAudio denoised = speechDenoiser.flush();
    appendSamples(output, denoised.getSamples());
    String outFilename = "enhanced-online-gtcrn.wav";
    WaveWriter.write(outFilename, toFloatArray(output), speechDenoiser.getSampleRate());
    System.out.printf("Saved to %s\n", outFilename);

    speechDenoiser.release();
  }
}
