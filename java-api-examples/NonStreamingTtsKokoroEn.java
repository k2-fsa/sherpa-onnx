// Copyright 2025 Xiaomi Corporation

// This file shows how to use a Kokoro English model
// to convert text to speech
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsKokoroEn {
  public static void main(String[] args) {
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
    // to download model files
    String model = "./kokoro-en-v0_19/model.onnx";
    String voices = "./kokoro-en-v0_19/voices.bin";
    String tokens = "./kokoro-en-v0_19/tokens.txt";
    String dataDir = "./kokoro-en-v0_19/espeak-ng-data";
    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsKokoroModelConfig kokoroModelConfig =
        OfflineTtsKokoroModelConfig.builder()
            .setModel(model)
            .setVoices(voices)
            .setTokens(tokens)
            .setDataDir(dataDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setKokoro(kokoroModelConfig)
            .setNumThreads(2)
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

    String waveFilename = "tts-kokoro-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
