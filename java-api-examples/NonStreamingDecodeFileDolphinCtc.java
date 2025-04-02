// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline Dolphin CTC model, i.e.,
// non-streaming Dolphin CTC model, to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileDolphinCtc {
  public static void main(String[] args) {
    // please refer to
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    // to download model files
    String model = "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/model.int8.onnx";
    String tokens = "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/tokens.txt";

    String waveFilename =
        "./sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineDolphinModelConfig dolphin = OfflineDolphinModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setDolphin(dolphin)
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
