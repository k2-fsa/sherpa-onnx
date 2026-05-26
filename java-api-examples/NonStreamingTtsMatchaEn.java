// Copyright 2025 Xiaomi Corporation

// This file shows how to use a matcha English model
// to convert text to speech
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsMatchaEn {
  public static void main(String[] args) {
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
    // to download model files
    String acousticModel = "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx";
    String vocoder = "./vocos-22khz-univ.onnx";
    String tokens = "./matcha-icefall-en_US-ljspeech/tokens.txt";
    String dataDir = "./matcha-icefall-en_US-ljspeech/espeak-ng-data";
    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsMatchaModelConfig matchaModelConfig =
        OfflineTtsMatchaModelConfig.builder()
            .setAcousticModel(acousticModel)
            .setVocoder(vocoder)
            .setTokens(tokens)
            .setDataDir(dataDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setMatcha(matchaModelConfig)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    GenerationConfig genConfig = new GenerationConfig();
    genConfig.setSid(0);
    genConfig.setSpeed(1.0f);
    genConfig.setSilenceScale(config.getSilenceScale());

    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generateWithConfigAndCallback(text, genConfig, (float[] samples) -> 1);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float realTimeFactor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-matcha-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- speaker ID: %d\n", genConfig.getSid());
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
