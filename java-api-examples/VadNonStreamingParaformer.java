// Copyright 2024 Xiaomi Corporation

// This file shows how to use a silero_vad model with a non-streaming Paraformer
// for speech recognition.

import com.k2fsa.sherpa.onnx.*;
import java.util.Arrays;

public class VadNonStreamingParaformer {
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
            .setWindowSize(512)
            .build();

    VadModelConfig config =
        VadModelConfig.builder()
            .setSileroVadModelConfig(sileroVad)
            .setSampleRate(16000)
            .setNumThreads(1)
            .setDebug(true)
            .setProvider("cpu")
            .build();

    return new Vad(config);
  }

  public static OfflineRecognizer createOfflineRecognizer() {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-03-28-chinese-english
    // to download model files
    String model = "./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx";
    String tokens = "./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt";

    String waveFilename = "./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/3-sichuan.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineParaformerModelConfig paraformer =
        OfflineParaformerModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setParaformer(paraformer)
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

    // You can download the test file from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    String testWaveFilename = "./lei-jun-test.wav";
    WaveReader reader = new WaveReader(testWaveFilename);

    int numSamples = reader.getSamples().length;
    int numIter = numSamples / 512;

    for (int i = 0; i != numIter; ++i) {
      int start = i * 512;
      int end = start + 512;
      float[] samples = Arrays.copyOfRange(reader.getSamples(), start, end);
      vad.acceptWaveform(samples);
      if (vad.isSpeechDetected()) {
        while (!vad.empty()) {
          SpeechSegment segment = vad.front();
          float startTime = segment.getStart() / 16000.0f;
          float duration = segment.getSamples().length / 16000.0f;

          OfflineStream stream = recognizer.createStream();
          stream.acceptWaveform(segment.getSamples(), 16000);
          recognizer.decode(stream);
          String text = recognizer.getResult(stream).getText();

          if (!text.isEmpty()) {
            System.out.printf("%.3f--%.3f: %s\n", startTime, startTime + duration, text);
          }

          vad.pop();
        }
      }
    }
  }
}
