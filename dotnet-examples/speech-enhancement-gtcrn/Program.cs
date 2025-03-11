// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use speech enhancement API with GTCRN models.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
//
// 2. Download a test file
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
//
// 3. Now run it
//
// dotnet run

using SherpaOnnx;

class OfflineSpeechEnhancementDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineSpeechDenoiserConfig();
    config.Model.Gtcrn.Model = "./gtcrn_simple.onnx";
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;
    var sd = new OfflineSpeechDenoiser(config);

    WaveReader waveReader = new WaveReader("./inp_16k.wav");
    var denoisedAudio =  sd.Run(waveReader.Samples, waveReader.SampleRate);

    var outputFilename = "./enhanced-16k.wav";
    var ok = denoisedAudio.SaveToWaveFile(outputFilename);

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
