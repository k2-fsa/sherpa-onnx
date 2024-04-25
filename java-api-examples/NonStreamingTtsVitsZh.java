// Copyright 2024 Xiaomi Corporation

// This file shows how to use a VITS Chinese TTS model
// to convert text to speech.
//
// You can use https://github.com/Plachtaa/VITS-fast-fine-tuning
// to train your model
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsPiperEn {
  public static void main(String[] args) {
    // please visit
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
    // to download model files
    String model = "./vits-zh-hf-fanchen-C/vits-zh-hf-fanchen-C.onnx";
    String tokens = "./vits-zh-hf-fanchen-C/tokens.txt";
    String lexicon = "./vits-zh-hf-fanchen-C/lexicon.txt";
    String dictDir = "./vits-zh-hf-fanchen-C/dict";
    String ruleFsts =
        "./vits-zh-hf-fanchen-C/phone.fst,./vits-zh-hf-fanchen-C/date.fst,./vits-zh-hf-fanchen-C/number.fst";
    String text = "有问题，请拨打110或者手机18601239876。我们的价值观是真诚热爱！";

    OfflineTtsVitsModelConfig vitsModelConfig =
        OfflineTtsVitsModelConfig.builder()
            .setModel(model)
            .setTokens(tokens)
            .setLexicon(lexicon)
            .setDictDir(dictDir)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setVits(vitsModelConfig)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    OfflineTtsConfig config =
        OfflineTtsConfig.builder().setModel(modelConfig).setRuleFsts(ruleFsts).build();

    OfflineTts tts = new OfflineTts(config);

    int sid = 100;
    float speed = 1.0f;
    long start = System.currentTimeMillis();
    GeneratedAudio audio = tts.generate(text, sid, speed);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-vits-zh.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
