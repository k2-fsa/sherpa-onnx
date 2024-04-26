// Copyright 2024 Xiaomi Corporation

// This file shows how to use a multilingual whisper model for
// spoken language identification.
//
// Note that it needs a multilingual whisper model. For instance,
// tiny works, but tiny.en doesn't.
import com.k2fsa.sherpa.onnx.*;

public class SpokenLanguageIdentificationWhisper {
  public static void main(String[] args) {
    // please download model and test files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    String encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx";
    String decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx";

    String[] testFiles =
        new String[] {
          "./spoken-language-identification-test-wavs/en-english.wav",
          "./spoken-language-identification-test-wavs/de-german.wav",
          "./spoken-language-identification-test-wavs/zh-chinese.wav",
          "./spoken-language-identification-test-wavs/es-spanish.wav",
          "./spoken-language-identification-test-wavs/fa-persian.wav",
          "./spoken-language-identification-test-wavs/ko-korean.wav",
          "./spoken-language-identification-test-wavs/ja-japanese.wav",
          "./spoken-language-identification-test-wavs/ru-russian.wav",
          "./spoken-language-identification-test-wavs/uk-ukrainian.wav",
        };

    SpokenLanguageIdentificationWhisperConfig whisper =
        SpokenLanguageIdentificationWhisperConfig.builder()
            .setEncoder(encoder)
            .setDecoder(decoder)
            .build();

    SpokenLanguageIdentificationConfig config =
        SpokenLanguageIdentificationConfig.builder()
            .setWhisper(whisper)
            .setNumThreads(1)
            .setDebug(true)
            .build();

    SpokenLanguageIdentification slid = new SpokenLanguageIdentification(config);
    for (String filename : testFiles) {
      WaveReader reader = new WaveReader(filename);

      OfflineStream stream = slid.createStream();
      stream.acceptWaveform(reader.getSamples(), reader.getSampleRate());

      String lang = slid.compute(stream);
      System.out.println("---");
      System.out.printf("filename: %s\n", filename);
      System.out.printf("lang: %s\n", lang);

      stream.release();
    }
    System.out.println("---");

    slid.release();
  }
}
