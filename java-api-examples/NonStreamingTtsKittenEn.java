// Copyright 2025 Xiaomi Corporation

// This file shows how to use a KittenTTS English model
// to convert text to speech
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsKittenEn {
  public static void main(String[] args) {
    LibraryUtils.enableDebug();
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kitten.html
    // to download model files
    String model = "./kitten-nano-en-v0_1-fp16/model.fp16.onnx";
    String voices = "./kitten-nano-en-v0_1-fp16/voices.bin";
    String tokens = "./kitten-nano-en-v0_1-fp16/tokens.txt";
    String dataDir = "./kitten-nano-en-v0_1-fp16/espeak-ng-data";
    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsKittenModelConfig kittenModelConfig =
        OfflineTtsKittenModelConfig.builder()
            .setModel(model)
            .setVoices(voices)
            .setTokens(tokens)
            .setDataDir(dataDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setKitten(kittenModelConfig)
            .setNumThreads(2)
            .setDebug(true)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    int sid = 7;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generate(text, sid, speed);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-kitten-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
