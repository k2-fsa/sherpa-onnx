// Copyright 2024 Xiaomi Corporation

// This file shows how to use a silero_vad model to detect speech
// and save detected speech into a wave file.

import com.k2fsa.sherpa.onnx.*;
import javax.sound.sampled.*;

public class VadFromMic {
  public static void main(String[] args) {
    int sampleRate = 16000;
    int windowSize = 512;
    // please download ./silero_vad.onnx from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    String model = "./silero_vad.onnx";
    SileroVadModelConfig sileroVad =
        SileroVadModelConfig.builder()
            .setModel(model)
            .setThreshold(0.5f)
            .setMinSilenceDuration(0.25f)
            .setMinSpeechDuration(0.5f)
            .setWindowSize(windowSize)
            .build();

    VadModelConfig config =
        VadModelConfig.builder()
            .setSileroVadModelConfig(sileroVad)
            .setSampleRate(sampleRate)
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .build();

    Vad vad = new Vad(config);

    // https://docs.oracle.com/javase/8/docs/api/javax/sound/sampled/AudioFormat.html
    // Linear PCM, 16000Hz, 16-bit, 1 channel, signed, little endian
    AudioFormat format = new AudioFormat(sampleRate, 16, 1, true, false);

    // https://docs.oracle.com/javase/8/docs/api/javax/sound/sampled/DataLine.Info.html#Info-java.lang.Class-javax.sound.sampled.AudioFormat-int-
    DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);
    TargetDataLine targetDataLine;
    try {
      targetDataLine = (TargetDataLine) AudioSystem.getLine(info);
      targetDataLine.open(format);
      targetDataLine.start();
    } catch (LineUnavailableException e) {
      System.out.println("Failed to open target data line: " + e.getMessage());
      vad.release();
      return;
    }

    boolean printed = false;
    int index = 0;

    byte[] buffer = new byte[windowSize * 2];
    float[] samples = new float[windowSize];

    while (targetDataLine.isOpen()) {
      int n = targetDataLine.read(buffer, 0, buffer.length);
      if (n <= 0) {
        System.out.printf("Got %d bytes. Expected %d bytes.\n", n, buffer.length);
        continue;
      }
      for (int i = 0; i != windowSize; ++i) {
        short low = buffer[2 * i];
        short high = buffer[2 * i + 1];
        int s = (high << 8) + low;
        samples[i] = (float) s / 32768;
      }

      vad.acceptWaveform(samples);
      if (vad.isSpeechDetected() && !printed) {
        System.out.println("Detected speech");
        printed = true;
      }

      if (!vad.isSpeechDetected()) {
        printed = false;
      }

      while (!vad.empty()) {
        float[] segment = vad.front().getSamples();
        float duration = segment.length / (float) sampleRate;
        System.out.printf("Duration: %.3f seconds\n", duration);

        String filename = String.format("seg-%d-%.3fs.wav", index, duration);
        index += 1;
        WaveWriter.write(filename, segment, sampleRate);
        System.out.printf("Saved to %s\n", filename);
        System.out.println("----------");
        vad.pop();
      }
    }

    vad.release();
  }
}
