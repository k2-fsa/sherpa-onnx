// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

// This file shows how to use an online transducer, i.e., streaming transducer,
// for real-time speech recognition with a microphone.
import com.k2fsa.sherpa.onnx.*;
import javax.sound.sampled.*;

public class StreamingAsrFromMicTransducer {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
    // to download model files
    String encoder =
        "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx";
    String decoder =
        "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx";
    String joiner =
        "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx";
    String tokens = "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt";

    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/itn_zh_number.fst
    String ruleFsts = "./itn_zh_number.fst";

    int sampleRate = 16000;

    OnlineTransducerModelConfig transducer =
        OnlineTransducerModelConfig.builder()
            .setEncoder(encoder)
            .setDecoder(decoder)
            .setJoiner(joiner)
            .build();

    OnlineModelConfig modelConfig =
        OnlineModelConfig.builder()
            .setTransducer(transducer)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OnlineRecognizerConfig config =
        OnlineRecognizerConfig.builder()
            .setOnlineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
            .setRuleFsts(ruleFsts)
            .build();

    OnlineRecognizer recognizer = new OnlineRecognizer(config);
    OnlineStream stream = recognizer.createStream();

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
      recognizer.release();
      stream.release();
      return;
    }

    String lastText = "";
    int segmentIndex = 0;

    // You can choose an arbitrary number
    int bufferSize = 1600; // 0.1 seconds for 16000Hz
    byte[] buffer = new byte[bufferSize * 2]; // a short has 2 bytes
    float[] samples = new float[bufferSize];

    System.out.println("Started! Please speak");
    while (targetDataLine.isOpen()) {
      int n = targetDataLine.read(buffer, 0, buffer.length);
      if (n <= 0) {
        System.out.printf("Got %d bytes. Expected %d bytes.\n", n, buffer.length);
        continue;
      }
      for (int i = 0; i != bufferSize; ++i) {
        short low = buffer[2 * i];
        short high = buffer[2 * i + 1];
        int s = (high << 8) + low;
        samples[i] = (float) s / 32768;
      }
      stream.acceptWaveform(samples, sampleRate);

      while (recognizer.isReady(stream)) {
        recognizer.decode(stream);
      }

      String text = recognizer.getResult(stream).getText();
      boolean isEndpoint = recognizer.isEndpoint(stream);
      if (!text.isEmpty() && text != " " && lastText != text) {
        lastText = text;
        System.out.printf("%d: %s\r", segmentIndex, text);
      }

      if (isEndpoint) {
        if (!text.isEmpty()) {
          System.out.println();
          segmentIndex += 1;
        }

        recognizer.reset(stream);
      }
    } // while (targetDataLine.isOpen())

    stream.release();
    recognizer.release();
  }
}
