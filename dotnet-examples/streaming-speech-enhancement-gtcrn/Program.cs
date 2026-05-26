// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the online speech enhancement API with GTCRN
// models.

using SherpaOnnx;

class StreamingSpeechEnhancementGtcrn
{
  static void Main(string[] args)
  {
    var config = new OnlineSpeechDenoiserConfig();
    config.Model.Gtcrn.Model = "./gtcrn_simple.onnx";
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;

    var sd = new OnlineSpeechDenoiser(config);
    WaveReader waveReader = new WaveReader("./inp_16k.wav");

    var samples = waveReader.Samples;
    var output = new List<float>(samples.Length);
    int frameShift = sd.FrameShiftInSamples;

    for (int start = 0; start < samples.Length; start += frameShift)
    {
      int count = Math.Min(frameShift, samples.Length - start);
      float[] chunk = new float[count];
      Array.Copy(samples, start, chunk, 0, count);
      var audio = sd.Run(chunk, waveReader.SampleRate);
      output.AddRange(audio.Samples);
    }

    output.AddRange(sd.Flush().Samples);

    var outFilename = "./enhanced-online-gtcrn.wav";
    var ok = DenoisedAudio.SaveToWaveFile(output.ToArray(), sd.SampleRate, outFilename);
    if (ok)
    {
      Console.WriteLine($"Wrote to {outFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outFilename}");
    }
  }
}
