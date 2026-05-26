// Copyright 2026 Xiaomi Corporation

// This file shows how to use an offline Qwen3 ASR model,
// i.e., non-streaming Qwen3 ASR model,
// to decode files.
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingDecodeFileQwen3Asr {
  public static void main(String[] args) {
    String modelDir = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25";
    String convFrontend = modelDir + "/conv_frontend.onnx";
    String encoder = modelDir + "/encoder.int8.onnx";
    String decoder = modelDir + "/decoder.int8.onnx";
    String tokenizer = modelDir + "/tokenizer";

    String tokens = "";

    String waveFilename = modelDir + "/test_wavs/raokouling.wav";

    WaveReader reader = new WaveReader(waveFilename);

    OfflineQwen3AsrModelConfig qwen3Asr =
        OfflineQwen3AsrModelConfig.builder()
            .setConvFrontend(convFrontend)
            .setEncoder(encoder)
            .setDecoder(decoder)
            .setTokenizer(tokenizer)
            .setHotwords("")
            .build();

    OfflineModelConfig modelConfig =
        OfflineModelConfig.builder()
            .setQwen3Asr(qwen3Asr)
            .setTokens(tokens)
            .setNumThreads(3)
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

    long start = System.currentTimeMillis();
    recognizer.decode(stream);
    long stop = System.currentTimeMillis();

    String text = recognizer.getResult(stream).getText();

    float timeElapsedSeconds = (stop - start) / 1000.0f;
    float audioDuration = reader.getSamples().length / (float) reader.getSampleRate();
    float realTimeFactor = timeElapsedSeconds / audioDuration;

    System.out.printf("filename:%s\nresult:%s\n", waveFilename, text);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);

    stream.release();
    recognizer.release();
  }
}
