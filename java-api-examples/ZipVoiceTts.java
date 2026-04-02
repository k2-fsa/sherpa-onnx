// Copyright 2026 Xiaomi Corporation

// This file shows how to use a ZipVoice Chinese/English model
// for zero-shot text to speech.
import com.k2fsa.sherpa.onnx.*;
import java.util.HashMap;
import java.util.Map;

public class ZipVoiceTts {
  public static void main(String[] args) {
    LibraryUtils.enableDebug();
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
    // to download model files
    String modelDir = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia";
    String referenceAudioFilename = modelDir + "/test_wavs/leijun-1.wav";
    String text = "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中.";
    String referenceText = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";

    OfflineTtsZipVoiceModelConfig zipvoiceModelConfig =
        OfflineTtsZipVoiceModelConfig.builder()
            .setTokens(modelDir + "/tokens.txt")
            .setEncoder(modelDir + "/encoder.int8.onnx")
            .setDecoder(modelDir + "/decoder.int8.onnx")
            .setVocoder("./vocos_24khz.onnx")
            .setDataDir(modelDir + "/espeak-ng-data")
            .setLexicon(modelDir + "/lexicon.txt")
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setZipvoice(zipvoiceModelConfig)
            .setNumThreads(2)
            .setDebug(false)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    WaveReader reader = new WaveReader(referenceAudioFilename);

    GenerationConfig genConfig = new GenerationConfig();
    genConfig.setReferenceAudio(reader.getSamples());
    genConfig.setReferenceSampleRate(reader.getSampleRate());
    genConfig.setReferenceText(referenceText);
    genConfig.setNumSteps(4);

    Map<String, String> extra = new HashMap<>();
    extra.put("min_char_in_sentence", "10");
    genConfig.setExtra(extra);

    long start = System.currentTimeMillis();
    GeneratedAudio audio =
        tts.generateWithConfigAndCallback(
            text,
            genConfig,
            samples -> {
              System.out.println("callback got called with " + samples.length + " samples");
              return 1;
            });
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;
    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float realTimeFactor = timeElapsedSeconds / audioDuration;

    String waveFilename = "generated-zipvoice-zh-en-java.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
