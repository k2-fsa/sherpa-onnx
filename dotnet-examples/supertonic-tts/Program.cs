// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming Supertonic TTS model
// for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using SherpaOnnx;
using System.Runtime.InteropServices;

class SupertonicTtsDemo
{
  static void Main(string[] args)
  {
    TestEn();
  }

  static void TestEn()
  {
    var config = new OfflineTtsConfig();
    config.Model.Supertonic.DurationPredictor = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/duration_predictor.int8.onnx";
    config.Model.Supertonic.TextEncoder = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/text_encoder.int8.onnx";
    config.Model.Supertonic.VectorEstimator = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vector_estimator.int8.onnx";
    config.Model.Supertonic.Vocoder = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/vocoder.int8.onnx";
    config.Model.Supertonic.TtsJson = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/tts.json";
    config.Model.Supertonic.UnicodeIndexer = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/unicode_indexer.bin";
    config.Model.Supertonic.VoiceStyle = "./sherpa-onnx-supertonic-tts-int8-2026-03-06/voice.bin";

    config.Model.NumThreads = 2;
    config.Model.Debug = 1;
    config.Model.Provider = "cpu";

    OfflineTtsGenerationConfig genConfig = new OfflineTtsGenerationConfig();
    genConfig.Sid = 6;
    genConfig.NumSteps = 5;
    genConfig.Speed = 1.25f;  // larger -> faster
    genConfig.Extra["lang"] = "en";

    var tts = new OfflineTts(config);
    var text = "Today as always, men fall into two groups: slaves and free men. Whoever " +
      "does not have two-thirds of his day for himself, is a slave, whatever " +
      "he may be: a statesman, a businessman, an official, or a scholar.";

    var MyCallback = (IntPtr samples, int n, float progress, IntPtr arg) =>
    {
      float[] data = new float[n];
      Marshal.Copy(samples, data, 0, n);
      // You can process samples here, e.g., play them.
      Console.WriteLine($"Progress {progress*100}%");

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var callback = new OfflineTtsCallbackProgressWithArg(MyCallback);

    var audio = tts.GenerateWithConfig(text, genConfig, callback);

    var outputFilename = "./generated-supertonic-en.wav";
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
