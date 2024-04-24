// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline transducer, i.e., non-streaming transducer,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileTransducer {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#sherpa-onnx-zipformer-gigaspeech-2023-12-12-english
    // to download model files
    String encoder =
        "./sherpa-onnx-zipformer-gigaspeech-2023-12-12/encoder-epoch-30-avg-1.int8.onnx";
    String decoder = "./sherpa-onnx-zipformer-gigaspeech-2023-12-12/decoder-epoch-30-avg-1.onnx";
    String joiner = "./sherpa-onnx-zipformer-gigaspeech-2023-12-12/joiner-epoch-30-avg-1.onnx";
    String tokens = "./sherpa-onnx-zipformer-gigaspeech-2023-12-12/tokens.txt";

    String waveFilename =
        "./sherpa-onnx-zipformer-gigaspeech-2023-12-12/test_wavs/1089-134686-0001.wav";

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
            .build();

    OfflineRecognizerConfig config =
        OfflineRecognizerConfig.builder()
            .setOfflineModelConfig(modelConfig)
            .setDecodingMethod("greedy_search")
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
