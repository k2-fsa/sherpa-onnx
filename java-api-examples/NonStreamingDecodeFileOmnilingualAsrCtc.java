// Copyright 2025 Xiaomi Corporation

// This file shows how to use an offline Omnilingual ASR CTC model,
// i.e., non-streaming Omnilingual ASR CTC model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileOmnilingualAsrCtc {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/sense-voice/index.html
    // to download model files
    String model =
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/model.int8.onnx";

    String tokens =
        "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/tokens.txt";

    String waveFilename =
        "./sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12/test_wavs/en.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineOmnilingualAsrCtcModelConfig omnilingual =
        OfflineOmnilingualAsrCtcModelConfig.builder().setModel(model).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setOmnilingual(omnilingual)
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
