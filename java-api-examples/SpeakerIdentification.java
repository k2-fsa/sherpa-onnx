// Copyright 2024 Xiaomi Corporation

// This file shows how to use a speaker embedding extractor model for speaker
// identification.
import com.k2fsa.sherpa.onnx.*;

public class SpeakerIdentification {
  public static float[] computeEmbedding(SpeakerEmbeddingExtractor extractor, String filename) {
    WaveReader reader = new WaveReader(filename);

    OnlineStream stream = extractor.createStream();
    stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());
    stream.inputFinished();

    float[] embedding = extractor.compute(stream);
    stream.release();

    return embedding;
  }

  public static void main(String[] args) {
    // Please download the model from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
    String model = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
    SpeakerEmbeddingExtractorConfig config =
        SpeakerEmbeddingExtractorConfig.builder()
            .setModel(model)
            .setNumThreads(1)
            .setDebug(true)
            .build();
    SpeakerEmbeddingExtractor extractor = new SpeakerEmbeddingExtractor(config);
    SpeakerEmbeddingManager manager = new SpeakerEmbeddingManager(extractor.getDim());

    String[] spk1Files =
        new String[] {
          "./sr-data/enroll/fangjun-sr-1.wav",
          "./sr-data/enroll/fangjun-sr-2.wav",
          "./sr-data/enroll/fangjun-sr-3.wav",
        };

    float[][] spk1Vec = new float[spk1Files.length][];

    for (int i = 0; i < spk1Files.length; ++i) {
      spk1Vec[i] = computeEmbedding(extractor, spk1Files[i]);
    }

    String[] spk2Files =
        new String[] {
          "./sr-data/enroll/leijun-sr-1.wav", "./sr-data/enroll/leijun-sr-2.wav",
        };

    float[][] spk2Vec = new float[spk2Files.length][];

    for (int i = 0; i < spk2Files.length; ++i) {
      spk2Vec[i] = computeEmbedding(extractor, spk2Files[i]);
    }

    if (!manager.add("fangjun", spk1Vec)) {
      System.out.println("Failed to register fangjun");
      return;
    }

    if (!manager.add("leijun", spk2Vec)) {
      System.out.println("Failed to register leijun");
      return;
    }

    if (manager.getNumSpeakers() != 2) {
      System.out.println("There should be two speakers");
      return;
    }

    if (!manager.contains("fangjun")) {
      System.out.println("It should contain the speaker fangjun");
      return;
    }

    if (!manager.contains("leijun")) {
      System.out.println("It should contain the speaker leijun");
      return;
    }

    System.out.println("---All speakers---");
    String[] allSpeakers = manager.getAllSpeakerNames();
    for (String s : allSpeakers) {
      System.out.println(s);
    }
    System.out.println("------------");

    String[] testFiles =
        new String[] {
          "./sr-data/test/fangjun-test-sr-1.wav",
          "./sr-data/test/leijun-test-sr-1.wav",
          "./sr-data/test/liudehua-test-sr-1.wav"
        };

    float threshold = 0.6f;
    for (String file : testFiles) {
      float[] embedding = computeEmbedding(extractor, file);

      String name = manager.search(embedding, threshold);
      if (name.isEmpty()) {
        name = "<Unknown>";
      }
      System.out.printf("%s: %s\n", file, name);
    }

    // test verify
    if (!manager.verify("fangjun", computeEmbedding(extractor, testFiles[0]), threshold)) {
      System.out.printf("testFiles[0] should match fangjun!");
      return;
    }

    if (!manager.remove("fangjun")) {
      System.out.println("Failed to remove fangjun");
      return;
    }

    if (manager.verify("fangjun", computeEmbedding(extractor, testFiles[0]), threshold)) {
      System.out.printf("%s should match no one!\n", testFiles[0]);
      return;
    }

    if (manager.getNumSpeakers() != 1) {
      System.out.println("There should only 1 speaker left.");
      return;
    }

    extractor.release();
    manager.release();
  }
}
