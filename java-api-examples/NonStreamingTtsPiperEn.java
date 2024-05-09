// Copyright 2024 Xiaomi Corporation

// This file shows how to use a piper VITS English TTS model
// to convert text to speech
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsPiperEn {
  public static void main(String[] args) {
    // please visit
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
    // to download model files
    String model = "./vits-piper-en_GB-cori-medium/en_GB-cori-medium.onnx";
    String tokens = "./vits-piper-en_GB-cori-medium/tokens.txt";
    String dataDir = "./vits-piper-en_GB-cori-medium/espeak-ng-data";
    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsVitsModelConfig vitsModelConfig =
        OfflineTtsVitsModelConfig.builder()
            .setModel(model)
            .setTokens(tokens)
            .setDataDir(dataDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setVits(vitsModelConfig)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    int sid = 0;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generate(text, sid, speed);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-piper-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
