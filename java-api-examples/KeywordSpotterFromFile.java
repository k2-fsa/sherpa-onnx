// Copyright 2024 Xiaomi Corporation

// This file shows how to use a keyword spotter model to spot keywords from
// a file.

import com.k2fsa.sherpa.onnx.*;

public class KyewordSpotterFromFile {
  public static void main(String[] args) {
    // please download test files from https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
    String encoder =
        "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx";
    String decoder =
        "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx";
    String joiner =
        "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx";
    String tokens = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt";

    String keywordsFile =
        "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt";

    String waveFilename = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav";

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

    KeywordSpotterConfig config =
        KeywordSpotterConfig.builder()
            .setOnlineModelConfig(modelConfig)
            .setKeywordsFile(keywordsFile)
            .build();

    KeywordSpotter kws = new KeywordSpotter(config);
    OnlineStream stream = kws.createStream();

    WaveReader reader = new WaveReader(waveFilename);

    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    float[] tailPaddings = new float[(int) (0.8 * reader.getSampleRate())];
    stream.acceptWaveform(tailPaddings, reader.getSampleRate());
    while (kws.isReady(stream)) {
      kws.decode(stream);

      String keyword = kws.getResult(stream).getKeyword();
      if (!keyword.isEmpty()) {
        System.out.printf("Detected keyword: %s\n", keyword);
      }
    }

    kws.release();
  }
}
