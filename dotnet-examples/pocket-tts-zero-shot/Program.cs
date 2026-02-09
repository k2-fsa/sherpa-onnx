// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming PocketTTS model
// for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pocket.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using SherpaOnnx;
using System.Runtime.InteropServices;

class PocketTtsDemo
{
  static void Main(string[] args)
  {

    TestEn();
  }

  static void TestEn()
  {
    var config = new OfflineTtsConfig();
    config.Model.Pocket.LmFlow = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx";
    config.Model.Pocket.LmMain = "./sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx";
    config.Model.Pocket.Encoder = "./sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx";
    config.Model.Pocket.Decoder = "./sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx";
    config.Model.Pocket.TextConditioner = "./sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx";
    config.Model.Pocket.VocabJson = "./sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json";
    config.Model.Pocket.TokenScoresJson = "./sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json";

    config.Model.NumThreads = 2;
    config.Model.Debug = 1;
    config.Model.Provider = "cpu";

    OfflineTtsGenerationConfig genConfig = new OfflineTtsGenerationConfig();

    var referenceWaveFilename = "./sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav";
    var reader = new WaveReader(referenceWaveFilename);

    genConfig.ReferenceAudio = reader.Samples;
    genConfig.ReferenceSampleRate = reader.SampleRate;
    genConfig.Extra["max_reference_audio_len"] = 12;

    var tts = new OfflineTts(config);
    var text = "Today as always, men fall into two groups: slaves and free men. Whoever " +
      "does not have two-thirds of his day for himself, is a slave, whatever " +
      "he may be: a statesman, a businessman, an official, or a scholar. " +
      "Friends fell out often because life was changing so fast. The easiest " +
      "thing in the world was to lose touch with someone.";

    var MyCallback = (IntPtr samples, int n, float progress, IntPtr arg) =>
    {
      float[] data = new float[n];
      Marshal.Copy(samples, data, 0, n);
      // You can process samples here, e.g., play them.
      // See ../kitten-tts-playback for how to play them
      Console.WriteLine($"Progress {progress*100}%");

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var callback = new OfflineTtsCallbackProgressWithArg(MyCallback);

    var audio = tts.GenerateWithConfig(text, genConfig, callback);

    var outputFilename = "./generated-pocket-en.wav";
    var ok = audio.SaveToWaveFile(outputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {outputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outputFilename}");
    }
  }
}

