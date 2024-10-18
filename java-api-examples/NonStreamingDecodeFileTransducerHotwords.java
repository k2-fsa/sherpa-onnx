// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline transducer, i.e., non-streaming transducer,
// to decode files with hotwords support.
//
// See also
// https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html#modeling-unit-is-cjkchar
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileTransducerHotwords {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html#modeling-unit-is-cjkchar
    // to download model files
    String encoder =
        "./sherpa-onnx-conformer-zh-stateless2-2023-05-23/encoder-epoch-99-avg-1.int8.onnx";
    String decoder = "./sherpa-onnx-conformer-zh-stateless2-2023-05-23/decoder-epoch-99-avg-1.onnx";
    String joiner = "./sherpa-onnx-conformer-zh-stateless2-2023-05-23/joiner-epoch-99-avg-1.onnx";
    String tokens = "./sherpa-onnx-conformer-zh-stateless2-2023-05-23/tokens.txt";

    String waveFilename = "./sherpa-onnx-conformer-zh-stateless2-2023-05-23/test_wavs/6.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineTransducerModelConfig transducer =
        OfflineTransducerModelConfig.builder()
            .setEncoder(encoder)
            .setDecoder(decoder)
            .setJoiner(joiner)
            .build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setTransducer(transducer)
            .setTokens(tokens)
            .setNumThreads(1)
            .setDebug(true)
            .setModelingUnit("cjkchar")
            .build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("modified_beam_search")
            .setHotwordsFile("./hotwords_cn.txt")
            .setHotwordsScore(2.0f)
            .build();

    OfflineRecognizer recognizer = new OfflineRecognizer(config);
    OfflineStream stream = recognizer.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    recognizer.decode(stream);

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}
