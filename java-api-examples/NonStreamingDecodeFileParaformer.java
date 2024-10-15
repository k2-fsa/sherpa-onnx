// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline paraformer, i.e., non-streaming paraformer,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileParaformer {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-09-14-chinese-english
    // to download model files
    String model = "./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx";
    String tokens = "./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt";

    String waveFilename = "./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/3-sichuan.wav";

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
