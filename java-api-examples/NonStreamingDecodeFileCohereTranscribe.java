// Copyright 2026 Xiaomi Corporation

// This file shows how to use an offline Cohere Transcribe model
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileCohereTranscribe {
  public static void main(String[] args) {
    String modelDir = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01";
    String encoder = modelDir + "/encoder.int8.onnx";
    String decoder = modelDir + "/decoder.int8.onnx";
    String tokens = modelDir + "/tokens.txt";

    String waveFilename = modelDir + "/test_wavs/en.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineCohereTranscribeModelConfig cohereTranscribe =
        OfflineCohereTranscribeModelConfig.builder()
            .setEncoder(encoder)
            .setDecoder(decoder)
            .setUsePunct(true)
            .setUseItn(true)
            .build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setCohereTranscribe(cohereTranscribe)
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
    stream.setOption("language", "en");
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

    recognizer.decode(stream);

    String text = recognizer.getResult(stream).getText();

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);

    stream.release();
    recognizer.release();
  }
}
