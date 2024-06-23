// Copyright 2024 Xiaomi Corporation

// This file shows how to use a silero_vad model with a non-streaming Whisper tiny.en
// for speech recognition.

import com.k2fsa.sherpa.onnx.*;
import javax.sound.sampled.*;

public class VadFromMicNonStreamingWhisper {
  private static final int sampleRate = 16000;
  private static final int windowSize = 512;

  public static Vad createVad() {
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

    return new Vad(config);
  }

  public static OfflineRecognizer createOfflineRecognizer() {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html
    // to download model files
    String encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx";
    String decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx";
    String tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt";

    OfflineWhisperModelConfig whisper =
        OfflineWhisperModelConfig.builder().setEncoder(encoder).setDecoder(decoder).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setWhisper(whisper)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
            .build();

    return new OfflineRecognizer(config);
  }

  public static void main(String[] args) {
    Vad vad = createVad();
    OfflineRecognizer recognizer = createOfflineRecognizer();

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
      recognizer.release();
      return;
    }

    boolean printed = false;
    byte[] buffer = new byte[windowSize * 2];
    float[] samples = new float[windowSize];

    System.out.println("Started. Please speak");
    boolean running = true;
    while (targetDataLine.isOpen() && running) {
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
        SpeechSegment segment = vad.front();
        float startTime = segment.getStart() / (float) sampleRate;
        float duration = segment.getSamples().length / (float) sampleRate;

        OfflineStream stream = recognizer.createStream();
        stream.acceptWaveform(segment.getSamples(), sampleRate);
        recognizer.decode(stream);
        String text = recognizer.getResult(stream).getText();
        stream.release();

        if (!text.isEmpty()) {
          System.out.printf("%.3f--%.3f: %s\n", startTime, startTime + duration, text);
        }

        if (text.contains("exit the program")) {
          running = false;
        }

        vad.pop();
      }
    }

    vad.release();
    recognizer.release();
  }
}
