// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

// This file shows how to use an online paraformer, i.e., streaming paraformer,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class StreamingDecodeFileParaformer {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-streaming-paraformer-bilingual-zh-en-chinese-english
    // to download model files
    String encoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx";
    String decoder = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx";
    String tokens = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt";
    String waveFilename = "./sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/2.wav";

    WaveReader reader = new WaveReader(waveFilename);
    System.out.println(reader.getSampleRate());
    System.out.println(reader.getSamples().length);

    OnlineParaformerModelConfig paraformer =
        OnlineParaformerModelConfig.builder().setEncoder(encoder).setDecoder(decoder).build();

    OnlineModelConfig modelConfig =
        OnlineModelConfig.builder()
            .setParaformer(paraformer)
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
