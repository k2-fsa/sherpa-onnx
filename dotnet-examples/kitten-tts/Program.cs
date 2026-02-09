// Copyright (c)  2025  Xiaomi Corporation
//
// This file shows how to use a non-streaming KittenTTS model
// for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using SherpaOnnx;
using System.Runtime.InteropServices;

class KittenTtsDemo
{
  static void Main(string[] args)
  {

    TestEn();
  }

  static void TestEn()
  {
    var config = new OfflineTtsConfig();
    config.Model.Kitten.Model = "./kitten-nano-en-v0_1-fp16/model.fp16.onnx";
    config.Model.Kitten.Voices = "./kitten-nano-en-v0_1-fp16/voices.bin";
    config.Model.Kitten.Tokens = "./kitten-nano-en-v0_1-fp16/tokens.txt";
    config.Model.Kitten.DataDir = "./kitten-nano-en-v0_1-fp16/espeak-ng-data";

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
    // 0->expr-voice-2-m, 1->expr-voice-2-f, 2->expr-voice-3-m
    // 3->expr-voice-3-f, 4->expr-voice-4-m, 5->expr-voice-4-f
    // 6->expr-voice-5-m, 7->expr-voice-5-f
    var sid = 0;

    var MyCallback = (IntPtr samples, int n, float progress) =>
    {
      float[] data = new float[n];
      Marshal.Copy(samples, data, 0, n);
      // You can process samples here, e.g., play them.
      // See ../kitten-tts-play for how to play them
      Console.WriteLine($"Progress {progress*100}%");

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var callback = new OfflineTtsCallbackProgress(MyCallback);

    var audio = tts.GenerateWithCallbackProgress(text, speed, sid, callback);

    var outputFilename = "./generated-kitten-en.wav";
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

