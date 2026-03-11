// Copyright 2026 Xiaomi Corporation

// This file shows how to use a Supertonic TTS English model.
import com.k2fsa.sherpa.onnx.*;
import java.util.HashMap;
import java.util.Map;

public class SupertonicTts {
  public static void main(String[] args) {
    LibraryUtils.enableDebug();
    // please visit
    // https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
    // to download model files
    String modelDir = "./sherpa-onnx-supertonic-tts-int8-2026-03-06";
    String durationPredictor = modelDir + "/duration_predictor.int8.onnx";
    String textEncoder = modelDir + "/text_encoder.int8.onnx";
    String vectorEstimator = modelDir + "/vector_estimator.int8.onnx";
    String vocoder = modelDir + "/vocoder.int8.onnx";
    String ttsJson = modelDir + "/tts.json";
    String unicodeIndexer = modelDir + "/unicode_indexer.bin";
    String voiceStyle = modelDir + "/voice.bin";

    String text =
        "Today as always, men fall into two groups: slaves and free men. Whoever does not have"
            + " two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a"
            + " businessman, an official, or a scholar.";

    OfflineTtsSupertonicModelConfig supertonicModelConfig =
        OfflineTtsSupertonicModelConfig.builder()
            .setDurationPredictor(durationPredictor)
            .setTextEncoder(textEncoder)
            .setVectorEstimator(vectorEstimator)
            .setVocoder(vocoder)
            .setTtsJson(ttsJson)
            .setUnicodeIndexer(unicodeIndexer)
            .setVoiceStyle(voiceStyle)
            .build();

    OfflineTtsModelConfig modelConfig =
        OfflineTtsModelConfig.builder()
            .setSupertonic(supertonicModelConfig)
            .setNumThreads(2)
            .setDebug(true)
            .build();

    OfflineTtsConfig config = OfflineTtsConfig.builder().setModel(modelConfig).build();
    OfflineTts tts = new OfflineTts(config);

    GenerationConfig genConfig = new GenerationConfig();
    genConfig.setSid(6);
    genConfig.setSpeed(1.25f);
    genConfig.setNumSteps(5);

    Map<String, String> extra = new HashMap<>();
    extra.put("lang", "en");

    genConfig.setExtra(extra);

    long start = System.currentTimeMillis();
    GeneratedAudio audio =
        tts.generateWithConfigAndCallback(
            text,
            genConfig,
            new OfflineTtsCallback() {
              @Override
              public Integer invoke(float[] samples) {
                System.out.println("callback got called with " + samples.length + " samples");
                return 1;
              }
            });

    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float realTimeFactor = timeElapsedSeconds / audioDuration;

    String waveFilename = "supertonic-tts-en.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
