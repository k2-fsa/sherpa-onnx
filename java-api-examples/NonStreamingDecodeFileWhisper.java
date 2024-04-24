// Copyright 2024 Xiaomi Corporation

// This file shows how to use an offline whisper, i.e., non-streaming whisper,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileWhisper {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html
    // to download model files
    String encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx";
    String decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx";
    String tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt";

    String waveFilename = "./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineWhisperModelConfig whisper =
        OfflineWhisperModelConfig.builder().setEncoder(encoder).setDecoder(decoder).build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setWhisper(whisper)
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
