// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline Zipformer CTC model,
// i.e., non-streaming Zipformer CTC model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileZipformerCtc {
  public static void main(String[] args) {
    // please refer to
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03.tar.bz2
    // to download model files
    String model = "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/model.int8.onnx";
    String tokens = "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/tokens.txt";

    String waveFilename = "./sherpa-onnx-zipformer-ctc-zh-int8-2025-07-03/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineZipformerCtcModelConfig zipformerCtc =
        OfflineZipformerCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setZipformerCtc(zipformerCtc)
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
