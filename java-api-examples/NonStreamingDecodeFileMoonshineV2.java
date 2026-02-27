// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline Moonshine,
// i.e., non-streaming Moonshine v2 model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileMoonshineV2 {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/moonshine/index.html
    // to download model files

    String encoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort";
    String decoder =
        "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort";
    String tokens = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt";

    String waveFilename = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineMoonshineModelConfig moonshine =
        OfflineMoonshineModelConfig.builder().setEncoder(encoder).setMergedDecoder(decoder).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setMoonshine(moonshine)
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
