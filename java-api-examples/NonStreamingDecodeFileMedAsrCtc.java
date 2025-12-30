// Copyright 2025 Xiaomi Corporation

// This file shows how to use an offline Google MedASR CTC model,
// i.e., non-streaming MedASR CTC model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileMedAsrCtc {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/medasr/index.html
    // to download model files
    String model = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/model.int8.onnx";

    String tokens = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/tokens.txt";

    String waveFilename = "./sherpa-onnx-medasr-ctc-en-int8-2025-12-25/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineMedAsrCtcModelConfig medasr =
        OfflineMedAsrCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setMedAsr(medasr)
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
