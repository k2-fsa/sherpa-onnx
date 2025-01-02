// Copyright 2025 Xiaomi Corporation

// This file shows how to use a matcha Chinese TTS model
// to convert text to speech
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsMatchaZh {
  public static void main(String[] args) {
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/matcha.html#matcha-icefall-zh-baker-chinese-1-female-speaker
    // to download model files
    String acousticModel = "./matcha-icefall-zh-baker/model-steps-3.onnx";
    String vocoder = "./hifigan_v2.onnx";
    String tokens = "./matcha-icefall-zh-baker/tokens.txt";
    String lexicon = "./matcha-icefall-zh-baker/lexicon.txt";
    String dictDir = "./matcha-icefall-zh-baker/dict";
    String ruleFsts =
        "./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst";
    String text =
        "某某银行的副行长和一些行政领导表示，他们去过长江"
            + "和长白山; 经济不断增长。"
            + "2024年12月31号，拨打110或者18920240511。"
            + "123456块钱。";

    OfflineTtsMatchaModelConfig matchaModelConfig =
        OfflineTtsMatchaModelConfig.builder()
            .setAcousticModel(acousticModel)
            .setVocoder(vocoder)
            .setTokens(tokens)
            .setLexicon(lexicon)
            .setDictDir(dictDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setMatcha(matchaModelConfig)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OfflineTtsConfig config =
        OfflineTtsConfig.builder().setModel(modelConfig).setRuleFsts(ruleFsts).build();
    OfflineTts tts = new OfflineTts(config);

    int sid = 0;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generate(text, sid, speed);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-matcha-zh.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
