// Copyright 2025 Xiaomi Corporation

// This file shows how to use an offline FireRedAsr AED model
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileFireRedAsr {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/FireRedAsr/index.html
    // to download model files
    String encoder = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/encoder.int8.onnx";
    String decoder = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/decoder.int8.onnx";
    String tokens = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/tokens.txt";

    String waveFilename = "./sherpa-onnx-fire-red-asr-large-zh_en-2025-02-16/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineFireRedAsrModelConfig fireRedAsr =
        OfflineFireRedAsrModelConfig.builder().setEncoder(encoder).setDecoder(decoder).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setFireRedAsr(fireRedAsr)
            .setTokens(tokens)
            .setNumThreads(2)
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
