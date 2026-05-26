// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming ZipVoice model
// for zero-shot text-to-speech.
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using SherpaOnnx;
using System.Runtime.InteropServices;

class ZipVoiceTtsDemo
{
  static void Main(string[] args)
  {
    TestZhEn();
  }

  static void TestZhEn()
  {
    var config = new OfflineTtsConfig();
    config.Model.ZipVoice.Tokens = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/tokens.txt";
    config.Model.ZipVoice.Encoder = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/encoder.int8.onnx";
    config.Model.ZipVoice.Decoder = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/decoder.int8.onnx";
    config.Model.ZipVoice.Vocoder = "./vocos_24khz.onnx";
    config.Model.ZipVoice.DataDir = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/espeak-ng-data";
    config.Model.ZipVoice.Lexicon = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/lexicon.txt";

    config.Model.NumThreads = 2;
    config.Model.Debug = 1;
    config.Model.Provider = "cpu";

    var referenceWaveFilename = "./sherpa-onnx-zipvoice-distill-int8-zh-en-emilia/test_wavs/leijun-1.wav";
    var reader = new WaveReader(referenceWaveFilename);

    OfflineTtsGenerationConfig genConfig = new OfflineTtsGenerationConfig();
    genConfig.ReferenceAudio = reader.Samples;
    genConfig.ReferenceSampleRate = reader.SampleRate;
    genConfig.ReferenceText = "那还是三十六年前, 一九八七年. 我呢考上了武汉大学的计算机系.";
    genConfig.NumSteps = 4;
    genConfig.Extra["min_char_in_sentence"] = "10";

    var tts = new OfflineTts(config);
    var text = "小米的价值观是真诚, 热爱. 真诚，就是不欺人也不自欺. 热爱, 就是全心投入并享受其中.";

    var myCallback = (IntPtr samples, int n, float progress, IntPtr arg) =>
    {
      float[] data = new float[n];
      Marshal.Copy(samples, data, 0, n);
      Console.WriteLine($"Progress {progress * 100}%");

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var callback = new OfflineTtsCallbackProgressWithArg(myCallback);

    var audio = tts.GenerateWithConfig(text, genConfig, callback);

    var outputFilename = "./generated-zipvoice-zh-en.wav";
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
