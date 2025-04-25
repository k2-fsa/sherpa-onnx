// Copyright 2025 Xiaomi Corporation

// This file shows how to use an offline whisper, i.e., non-streaming whisper,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileWhisperMultiple {
  public static void main(String[] args) {
    // please refer to
    // https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html
    // to download model files
    String encoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.int8.onnx";
    String decoder = "./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.int8.onnx";
    String tokens = "./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt";

    String waveFilename0 = "./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav";
    String waveFilename1 = "./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav";

    WaveReader reader0 = new WaveReader(waveFilename0);
    WaveReader reader1 = new WaveReader(waveFilename1);

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
    OfflineStream stream0 = recognizer.createStream();
    stream0.acceptWaveform(reader0.getSamples(), reader0.getSampleRate());

    OfflineStream stream1 = recognizer.createStream();
    stream1.acceptWaveform(reader1.getSamples(), reader1.getSampleRate());

    OfflineStream[] ss = new OfflineStream[] {stream0, stream1};
    recognizer.decode(ss);

    String text0 = recognizer.getResult(stream0).getText();
    String text1 = recognizer.getResult(stream1).getText();

    System.out.printf("filename0:%s\nresult0:%s\n\n", waveFilename0, text0);
    System.out.printf("filename1:%s\nresult1:%s\n\n", waveFilename1, text1);

    stream0.release();
    stream1.release();
    recognizer.release();
  }
}
