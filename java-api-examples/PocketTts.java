// Copyright 2026 Xiaomi Corporation

// This file shows how to use a PocketTTS English model
// for voice cloning.
import com.k2fsa.sherpa.onnx.*;
import java.util.HashMap;
import java.util.Map;

public class PocketTts {
  public static void main(String[] args) {
    LibraryUtils.enableDebug();
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
    // to download model files
    String lmFlow = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx";
    String lmMain = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx";
    String encoder = "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx";
    String decoder = "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx";
    String textConditioner = "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx";
    String vocabJson = "./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json";
    String tokenScoresJson = "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json";
    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsPocketModelConfig pocketModelConfig =
        OfflineTtsPocketModelConfig.builder()
            .setLmMain(lmMain)
            .setLmFlow(lmFlow)
            .setEncoder(encoder)
            .setDecoder(decoder)
            .setTextConditioner(textConditioner)
            .setVocabJson(vocabJson)
            .setTokenScoresJson(tokenScoresJson)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setPocket(pocketModelConfig)
            .setNumThreads(2)
            .setDebug(true)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    String referenceAudioFilename = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav";
    WaveReader reader = new WaveReader(referenceAudioFilename);

    GenerationConfig genConfig = new GenerationConfig();
    genConfig.setReferenceAudio(reader.getSamples());
    genConfig.setReferenceSampleRate(reader.getSampleRate());
    genConfig.setNumSteps(5);

    Map<String, String> extra = new HashMap<>();
    extra.put("temperature", "0.7");
    extra.put("chunk_size", "15");

    genConfig.setExtra(extra);

    long start = System.currentTimeMillis();
    GeneratedAudio audio = null;

    // You can choose one of the following callback style
    // ---------------------------------------------------
    // 1. Anonymous class implementing OfflineTtsCallback
    // ---------------------------------------------------
    if (true) {
      audio =
          tts.generateWithConfigAndCallback(
              text,
              genConfig,
              new OfflineTtsCallback() {
                @Override
                public Integer invoke(float[] samples) {
                  // you can play the generated samples in a separate thread
                  System.out.println("callback got called with " + samples.length + " samples");
                  // 1 = continue, 0 = stop
                  return 1;
                }
              });
    }

    // -------------------------------
    // 2. Lambda implementing OfflineTtsCallback
    // -------------------------------
    if (false) {
      audio =
          tts.generateWithConfigAndCallback(
              text,
              genConfig,
              samples -> {
                System.out.println("Lambda Integer callback: " + samples.length);
                return 1; // continue
              });
    }

    if (false) {
      audio =
          tts.generateWithConfigAndCallback(
              text,
              genConfig,
              samples -> {
                System.out.println("Consumer: " + samples.length);
                // implicitly, it returns 1 internally
              });
    }

    if (audio == null) {
      System.err.println("No audio was generated. Please enable at least one callback branch.");
      return;
    }

    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float real_time_factor = timeElapsedSeconds / audioDuration;

    String waveFilename = "pocket-tts-bria.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", real_time_factor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
