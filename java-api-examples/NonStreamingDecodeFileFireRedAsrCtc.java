// Copyright 2025 Xiaomi Corporation

// This file shows how to use an offline FireRedASR CTC model,
// i.e., non-streaming FireRedASR CTC model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileFireRedAsrCtc {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/FireRedAsr/index.html
    // to download model files
    String model = "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/model.int8.onnx";

    String tokens = "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/tokens.txt";

    String waveFilename = "./sherpa-onnx-fire-red-asr2-ctc-zh_en-int8-2026-02-25/test_wavs/1.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineFireRedAsrCtcModelConfig medasr =
        OfflineFireRedAsrCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setFireRedAsrCtc(medasr)
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
