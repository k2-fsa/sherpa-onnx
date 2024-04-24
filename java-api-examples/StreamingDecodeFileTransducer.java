// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

// This file shows how to use an online transducer, i.e., streaming transducer,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class StreamingDecodeFileTransducer {
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

    String waveFilename =
        "./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

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
            .build();

    OnlineRecognizer recognizer = new OnlineRecognizer(config);
    OnlineStream stream = recognizer.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    float[] tailPaddings = new float[(int) (0.8 * reader.getSampleRate())];
    stream.acceptWaveform(tailPaddings, reader.getSampleRate());

    while (recognizer.isReady(stream)) {
      recognizer.decode(stream);
    }

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}
