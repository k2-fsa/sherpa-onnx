// Copyright 2025 Xiaomi Corporation

// This file shows how to use a ten-vad model to remove silences from
// a wave file.

import com.k2fsa.sherpa.onnx.*;
import java.util.ArrayList;
import java.util.Arrays;

public class TenVadRemoveSilence {
  public static void main(String[] args) {
    // please download ./ten-vad.onnx from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    String model = "./ten-vad.onnx";
    TenVadModelConfig tenVad =
        TenVadModelConfig.builder()
            .setModel(model)
            .setThreshold(0.5f)
            .setMinSilenceDuration(0.25f)
            .setMinSpeechDuration(0.5f)
            .setWindowSize(256)
            .setMaxSpeechDuration(5.0f)
            .build();

    VadModelConfig config =
        VadModelConfig.builder()
            .setTenVadModelConfig(tenVad)
            .setSampleRate(16000)
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .build();

    Vad vad = new Vad(config);

    // You can download the test file from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    String testWaveFilename = "./lei-jun-test.wav";
    WaveReader reader = new WaveReader(testWaveFilename);

    int numSamples = reader.getSamples().length;
    int windowSize = tenVad.getWindowSize();
    int numIter = numSamples / windowSize;

    ArrayList<float[]> segments = new ArrayList<float[]>();

    for (int i = 0; i != numIter; ++i) {
      int start = i * windowSize;
      int end = start + windowSize;
      float[] samples = Arrays.copyOfRange(reader.getSamples(), start, end);
      vad.acceptWaveform(samples);
      if (vad.isSpeechDetected()) {
        while (!vad.empty()) {

          // if you want to get the starting time of this segment, you can use
          /* float startTime = vad.front().getStart() / 16000.0f; */

          segments.add(vad.front().getSamples());
          vad.pop();
        }
      }
    }

    vad.flush();
    while (!vad.empty()) {

      // if you want to get the starting time of this segment, you can use
      /* float startTime = vad.front().getStart() / 16000.0f; */

      segments.add(vad.front().getSamples());
      vad.pop();
    }

    // get total number of samples
    int n = 0;
    for (float[] s : segments) {
      n += s.length;
    }

    float[] allSamples = new float[n];
    int i = 0;
    for (float[] s : segments) {
      System.arraycopy(s, 0, allSamples, i, s.length);
      i += s.length;
    }

    String outFilename = "lei-jun-test-no-silence.wav";
    WaveWriter.write(outFilename, allSamples, 16000);
    System.out.printf("Saved to %s\n", outFilename);

    vad.release();
  }
}
