// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the speech enhancement API with VoxLock models.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock takes the loudest
// 3 seconds in the first 15 seconds of the input as the speaker to keep. See
// ../speech-enhancement-voxlock-with-voice-sample if you have a voice sample
// of the speaker.
//
// 1. Download the model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speech-enhancement-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
// tar xvf sherpa-onnx-voxlock-int8.tar.bz2
//
// 2. Now run it
//
// dotnet run

using System.Diagnostics;
using SherpaOnnx;

class SpeechEnhancementVoxLock
{
  static void Main(string[] args)
  {
    var config = new OfflineSpeechDenoiserConfig();
    config.Model.Voxlock.EnrollModel = "./sherpa-onnx-voxlock-int8/enroll.onnx";
    config.Model.Voxlock.ChunkModel = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx";
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;
    var sd = new OfflineSpeechDenoiser(config);

    WaveReader waveReader = new WaveReader("./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav");

    var stopwatch = Stopwatch.StartNew();
    var denoisedAudio = sd.Run(waveReader.Samples, waveReader.SampleRate);
    stopwatch.Stop();

    var outputFilename = "./voxlock-enhanced-16k.wav";
    var ok = denoisedAudio.SaveToWaveFile(outputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {outputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outputFilename}");
    }

    double elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
    double duration = waveReader.Samples.Length / (double)waveReader.SampleRate;
    Console.WriteLine($"Elapsed seconds: {elapsedSeconds:F3}");
    Console.WriteLine($"Audio duration in seconds: {duration:F3}");
    Console.WriteLine($"RTF: {elapsedSeconds:F3}/{duration:F3} = {elapsedSeconds / duration:F3}");
  }
}
