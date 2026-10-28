// Copyright 2026 Xiaomi Corporation

// This file shows how to use a matcha English TTS model to convert text to
// speech.
//
// The text is phonemized outside of sherpa-onnx with piper-phonemize
// (https://github.com/csukuangfj/piper-phonemize) and the resulting phoneme
// codepoints are passed via GenerationConfig.phonemeCodepoints. No lexicon
// file and no espeak-ng-data are needed.
// If you want to use a lexicon instead, please see
// NonStreamingTtsMatchaEnLexicon.java.
import com.github.csukuangfj.piper.phonemize.PiperPhonemize;
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsMatchaEnPhonemize {
  public static void main(String[] args) {
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-en-us-ljspeech-american-english-1-female-speaker
    // to download model files
    String acousticModel = "./matcha-icefall-en_US-ljspeech/model-steps-3.onnx";
    String vocoder = "./vocos-22khz-univ.onnx";
    String tokens = "./matcha-icefall-en_US-ljspeech/tokens.txt";
    String text =
                "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
                    + "two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.";

    OfflineTtsMatchaModelConfig matchaModelConfig =
        OfflineTtsMatchaModelConfig.builder()
            .setAcousticModel(acousticModel)
            .setVocoder(vocoder)
            .setTokens(tokens)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setMatcha(matchaModelConfig)
            .setNumThreads(1)
            .setDebug(true)
            .build();
    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    System.out.printf("piper-phonemize version: %s\n", PiperPhonemize.getVersionStr());
    // initialize() extracts the espeak-ng-data embedded in piper-phonemize.
    // That data only feeds the piper phonemizer; sherpa-onnx itself does not
    // use espeak-ng anymore.
    PiperPhonemize.initialize();

    // The espeak-ng voice for the phonemizer is taken from the model metadata.
    // Fall back to en-us if the model has no language information.
    String voice = tts.getLang();
    if (voice == null || voice.isEmpty()) {
      System.out.println("Warning: the model has no language information. Falling back to en-us.");
      voice = "en-us";
    }
    System.out.printf("Phonemizer language: %s\n", voice);

    int[][] phonemeCodepoints;
    try (PiperPhonemize.Result result = PiperPhonemize.text(text, voice)) {
      if (result == null) {
        throw new RuntimeException("Failed to phonemize the text");
      }
      phonemeCodepoints = new int[result.getNumSentences()][];
      for (int i = 0; i < result.getNumSentences(); i++) {
        phonemeCodepoints[i] = result.getPhonemes(i);
      }
    }
    System.out.printf("Number of phonemized sentences: %d\n", phonemeCodepoints.length);

    int sid = 0;
    float speed = 1.0f;
    GenerationConfig genConfig = new GenerationConfig();
    genConfig.setSid(sid);
    genConfig.setSpeed(speed);
    genConfig.setSilenceScale(config.getSilenceScale());
    genConfig.setPhonemeCodepoints(phonemeCodepoints);

    long start = System.currentTimeMillis();
    // The text has already been phonemized; pass an empty string here.
    GeneratedAudio audio = tts.generateWithConfig("", genConfig);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float realTimeFactor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-matcha-en-phonemize.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
