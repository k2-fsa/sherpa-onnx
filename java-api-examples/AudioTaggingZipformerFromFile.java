// Copyright 2024 Xiaomi Corporation

// This file shows how to use a zipformer audio tagging model to tag
// input audio files.
import com.k2fsa.sherpa.onnx.*;

public class AudioTaggingZipformerFromFile {
  public static void main(String[] args) {
    // please download the model from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
    String model = "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.int8.onnx";
    String labels =
        "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/class_labels_indices.csv";
    int topK = 5;

    String[] testWaves =
        new String[] {
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/1.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/2.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/3.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/4.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/5.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/6.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/7.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/8.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/9.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/10.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/11.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/12.wav",
          "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/13.wav",
        };

    OfflineZipformerAudioTaggingModelConfig zipformer =
        OfflineZipformerAudioTaggingModelConfig.builder().setModel(model).build();

    AudioTaggingModelConfig modelConfig =
        AudioTaggingModelConfig.builder()
            .setZipformer(zipformer)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    AudioTaggingConfig config =
        AudioTaggingConfig.builder().setModel(modelConfig).setLabels(labels).setTopK(topK).build();

    AudioTagging tagger = new AudioTagging(config);
    System.out.println("------");
    for (String filename : testWaves) {
      WaveReader reader = new WaveReader(filename);

      OfflineStream stream = tagger.createStream();
      stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

      AudioEvent[] events = tagger.compute(stream);

      stream.release();

      System.out.printf("input file: %s\n", filename);
      System.out.printf("Probability\t\tName\n");
      for (AudioEvent e : events) {
        System.out.printf("%.3f\t\t\t%s\n", e.getProb(), e.getName());
      }
      System.out.println("------");
    }

    tagger.release();
  }
}
