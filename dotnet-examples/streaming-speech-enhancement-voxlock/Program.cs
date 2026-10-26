// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the online speech enhancement API with VoxLock
// models.
//
// VoxLock is a target-speaker voice isolation model. It keeps the voice of one
// speaker, e.g., the caller in a phone call, and removes the other voices.
//
// The model is from https://github.com/arbazsiddiqui/voxlock
//
// This example does not provide a voice sample, so VoxLock passes the audio
// through unchanged until it has heard 3 seconds of speech, takes that speech
// as the speaker to keep, and isolates from then on. See
// ../streaming-speech-enhancement-voxlock-with-voice-sample if you have a
// voice sample of the speaker.

using System.Diagnostics;
using SherpaOnnx;

class StreamingSpeechEnhancementVoxLock
{
  static void Main(string[] args)
  {
    var config = new OnlineSpeechDenoiserConfig();
    config.Model.Voxlock.EnrollModel = "./sherpa-onnx-voxlock-int8/enroll.onnx";
    config.Model.Voxlock.ChunkModel = "./sherpa-onnx-voxlock-int8/chunk_int8.onnx";
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;

    var sd = new OnlineSpeechDenoiser(config);
    WaveReader waveReader = new WaveReader("./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav");

    var samples = waveReader.Samples;
    var output = new List<float>(samples.Length);
    int frameShift = sd.FrameShiftInSamples;

    var stopwatch = Stopwatch.StartNew();
    for (int start = 0; start < samples.Length; start += frameShift)
    {
      int count = Math.Min(frameShift, samples.Length - start);
      float[] chunk = new float[count];
      Array.Copy(samples, start, chunk, 0, count);
      var audio = sd.Run(chunk, waveReader.SampleRate);
      output.AddRange(audio.Samples);
    }

    output.AddRange(sd.Flush().Samples);
    stopwatch.Stop();

    var outFilename = "./voxlock-enhanced-online-16k.wav";
    var ok = DenoisedAudio.SaveToWaveFile(output.ToArray(), sd.SampleRate, outFilename);
    if (ok)
    {
      Console.WriteLine($"Wrote to {outFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outFilename}");
    }

    double elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
    double duration = samples.Length / (double)waveReader.SampleRate;
    Console.WriteLine($"Elapsed seconds: {elapsedSeconds:F3}");
    Console.WriteLine($"Audio duration in seconds: {duration:F3}");
    Console.WriteLine($"RTF: {elapsedSeconds:F3}/{duration:F3} = {elapsedSeconds / duration:F3}");
  }
}
