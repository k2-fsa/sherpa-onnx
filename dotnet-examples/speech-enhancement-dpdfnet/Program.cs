// Copyright (c)  2025  Xiaomi Corporation
//
// This file shows how to use speech enhancement API with DPDFNet models.
// Use dpdfnet_baseline.onnx, dpdfnet2.onnx, dpdfnet4.onnx, or dpdfnet8.onnx
// for 16 kHz downstream ASR or speech recognition.
// Use dpdfnet2_48khz_hr.onnx for 48 kHz enhancement output.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2.onnx
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet4.onnx
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet8.onnx
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet2_48khz_hr.onnx
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
    var model = "./dpdfnet_baseline.onnx";
    var config = new OfflineSpeechDenoiserConfig();
    config.Model.Dpdfnet.Model = model;
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;
    var sd = new OfflineSpeechDenoiser(config);

    WaveReader waveReader = new WaveReader("./inp_16k.wav");
    var denoisedAudio = sd.Run(waveReader.Samples, waveReader.SampleRate);

    var outputFilename = "./enhanced.wav";
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
