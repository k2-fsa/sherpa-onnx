// Copyright (c)  2025  Xiaomi Corporation
//
// This file shows how to use a non-streaming Kokoro TTS model
// for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using SherpaOnnx;
using System.Runtime.InteropServices;

class OfflineTtsDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineTtsConfig();
    config.Model.Kokoro.Model = "./kokoro-en-v0_19/model.onnx";
    config.Model.Kokoro.Voices = "./kokoro-en-v0_19/voices.bin";
    config.Model.Kokoro.Tokens = "./kokoro-en-v0_19/tokens.txt";
    config.Model.Kokoro.DataDir = "./kokoro-en-v0_19/espeak-ng-data";

    config.Model.NumThreads = 2;
    config.Model.Debug = 1;
    config.Model.Provider = "cpu";

    var tts = new OfflineTts(config);
    var speed = 1.0f;
    var text = "Today as always, men fall into two groups: slaves and free men. Whoever " +
      "does not have two-thirds of his day for himself, is a slave, whatever " +
      "he may be: a statesman, a businessman, an official, or a scholar. " +
      "Friends fell out often because life was changing so fast. The easiest " +
      "thing in the world was to lose touch with someone.";

    // mapping of sid to voice name
    // 0->af, 1->af_bella, 2->af_nicole, 3->af_sarah, 4->af_sky, 5->am_adam
    // 6->am_michael, 7->bf_emma, 8->bf_isabella, 9->bm_george, 10->bm_lewis
    var sid = 0;

    var MyCallback = (IntPtr samples, int n, float progress) =>
    {
      float[] data = new float[n];
      Marshal.Copy(samples, data, 0, n);
      // You can process samples here, e.g., play them.
      // See ../kokoro-tts-playback for how to play them
      Console.WriteLine($"Progress {progress*100}%");

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var callback = new OfflineTtsCallbackProgress(MyCallback);

    var audio = tts.GenerateWithCallbackProgress(text, speed, sid, callback);

    var outputFilename = "./generated-kokoro-0.wav";
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

