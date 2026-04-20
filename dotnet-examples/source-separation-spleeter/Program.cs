// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the source separation API with Spleeter models.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/sherpa-onnx-spleeter-2stems-fp16.tar.bz2
//
// tar xjf sherpa-onnx-spleeter-2stems-fp16.tar.bz2
//
// 2. Download a test file
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
//
// 3. Now run it
//
// dotnet run

using SherpaOnnx;
using System.Diagnostics;

class OfflineSourceSeparationDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineSourceSeparationConfig();
    config.Model.Spleeter.Vocals = "./sherpa-onnx-spleeter-2stems-fp16/vocals.fp16.onnx";
    config.Model.Spleeter.Accompaniment = "./sherpa-onnx-spleeter-2stems-fp16/accompaniment.fp16.onnx";
    config.Model.NumThreads = 1;

    var ss = new OfflineSourceSeparation(config);

    using var wave = OfflineSourceSeparation.ReadWaveMultiChannel("./qi-feng-le-zh.wav");
    if (wave == null)
    {
      Console.WriteLine("Failed to read ./qi-feng-le-zh.wav");
      return;
    }

    Console.WriteLine($"Input wave: channels={wave.NumChannels}, samples_per_channel={wave.NumSamples}, sample_rate={wave.SampleRate}");

    var channels = wave.GetChannels();
    Console.WriteLine("Started");
    var stopwatch = Stopwatch.StartNew();
    var output = ss.Process(channels, wave.SampleRate);
    stopwatch.Stop();
    Console.WriteLine("Done");

    Console.WriteLine($"Output: {output.NumStems} stems, sample_rate={output.SampleRate}");

    string[] stemNames = {"spleeter-vocals", "spleeter-accompaniment"};
    for (int s = 0; s < output.NumStems && s < stemNames.Length; ++s)
    {
      string filename = stemNames[s] + ".wav";
      bool ok = output.SaveStemToWaveFile(s, filename);
      if (ok)
      {
        Console.WriteLine($"Saved {filename}");
      }
      else
      {
        Console.WriteLine($"Failed to save {filename}");
      }
    }

    float elapsedSeconds = stopwatch.ElapsedMilliseconds / 1000.0f;
    float duration = wave.NumSamples / (float)wave.SampleRate;
    float rtf = elapsedSeconds / duration;

    Console.WriteLine($"Duration: {duration:F3}s");
    Console.WriteLine($"Elapsed seconds: {elapsedSeconds:F3}s");
    Console.WriteLine($"(Real time factor) RTF = {elapsedSeconds:F3} / {duration:F3} = {rtf:F3}");

    output.Dispose();
  }
}
