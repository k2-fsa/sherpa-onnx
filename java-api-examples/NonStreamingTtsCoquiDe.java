// Copyright 2024 Xiaomi Corporation

// This file shows how to use a Coqui-ai VITS German TTS model
// to convert text to speech
import com.k2fsa.sherpa.onnx.*;

public class NonStreamingTtsCoquiDe {
  public static void main(String[] args) {
    // please visit
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
    // to download model files
    String model = "./vits-coqui-de-css10/model.onnx";
    String tokens = "./vits-coqui-de-css10/tokens.txt";
    String text = "Alles hat ein Ende, nur die Wurst hat zwei.";

    OfflineTtsVitsModelConfig vitsModelConfig =
        OfflineTtsVitsModelConfig.builder().setModel(model).setTokens(tokens).build();

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
    GenerationConfig genConfig = new GenerationConfig();
    genConfig.setSid(sid);
    genConfig.setSpeed(speed);
    genConfig.setSilenceScale(config.getSilenceScale());
    long start = System.currentTimeMillis();
    GeneratedAudio audio =
        tts.generateWithConfigAndCallback(text, genConfig, (float[] samples) -> 1);
    long stop = System.currentTimeMillis();

    float timeElapsedSeconds = (stop - start) / 1000.0f;

    float audioDuration = audio.getSamples().length / (float) audio.getSampleRate();
    float realTimeFactor = timeElapsedSeconds / audioDuration;

    String waveFilename = "tts-coqui-de.wav";
    audio.save(waveFilename);
    System.out.printf("-- elapsed : %.3f seconds\n", timeElapsedSeconds);
    System.out.printf("-- audio duration: %.3f seconds\n", audioDuration);
    System.out.printf("-- real-time factor (RTF): %.3f\n", realTimeFactor);
    System.out.printf("-- text: %s\n", text);
    System.out.printf("-- Saved to %s\n", waveFilename);

    tts.release();
  }
}
