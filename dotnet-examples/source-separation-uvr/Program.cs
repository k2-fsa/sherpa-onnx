// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the source separation API with UVR (MDX-Net) models.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/UVR-MDX-NET-Voc_FT.onnx
//
// 2. Download a test file
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/source-separation-models/qi-feng-le-zh.wav
//
// 3. Now run it
//
// dotnet run

using SherpaOnnx;

class OfflineSourceSeparationDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineSourceSeparationConfig();
    config.Model.Uvr.Model = "./UVR-MDX-NET-Voc_FT.onnx";
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
    var output = ss.Process(channels, wave.SampleRate);

    Console.WriteLine($"Output: {output.NumStems} stems, sample_rate={output.SampleRate}");

    string[] stemNames = {"uvr-vocals", "uvr-non-vocals"};
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

    output.Dispose();
  }
}
