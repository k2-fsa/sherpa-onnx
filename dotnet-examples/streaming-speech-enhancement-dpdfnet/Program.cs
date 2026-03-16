// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use the online speech enhancement API with DPDFNet
// models.

using SherpaOnnx;
using System.Runtime.InteropServices;
using System.Text;

class StreamingSpeechEnhancementDpdfnet
{
  static void Main(string[] args)
  {
    var config = new OnlineSpeechDenoiserConfig();
    config.Model.Dpdfnet.Model = "./dpdfnet_baseline.onnx";
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
      output.AddRange(sd.Run(chunk, waveReader.SampleRate).Samples);
    }

    output.AddRange(sd.Flush().Samples);

    var outFilename = "./enhanced-online-dpdfnet.wav";
    var outAudio = new GeneratedDenoisedAudio(output.ToArray(), sd.SampleRate);
    if (outAudio.SaveToWaveFile(outFilename))
    {
      Console.WriteLine($"Wrote to {outFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outFilename}");
    }
  }

  private sealed class GeneratedDenoisedAudio
  {
    private readonly float[] _samples;
    private readonly int _sampleRate;

    public GeneratedDenoisedAudio(float[] samples, int sampleRate)
    {
      _samples = samples;
      _sampleRate = sampleRate;
    }

    public bool SaveToWaveFile(string filename)
    {
      byte[] utf8Filename = Encoding.UTF8.GetBytes(filename);
      byte[] utf8FilenameWithNull = new byte[utf8Filename.Length + 1];
      Array.Copy(utf8Filename, utf8FilenameWithNull, utf8Filename.Length);
      utf8FilenameWithNull[utf8Filename.Length] = 0;
      return SherpaOnnxWriteWave(_samples, _samples.Length, _sampleRate, utf8FilenameWithNull) == 1;
    }

    [DllImport(Dll.Filename)]
    private static extern int SherpaOnnxWriteWave(
        float[] samples,
        int n,
        int sampleRate,
        [MarshalAs(UnmanagedType.LPArray, ArraySubType = UnmanagedType.I1)] byte[] utf8Filename);
  }
}
