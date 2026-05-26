// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming ZipVoice model
// for zero-shot text-to-speech with playback.
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/zipvoice.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using PortAudioSharp;
using SherpaOnnx;
using System.Collections.Concurrent;
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

    Console.WriteLine(PortAudio.VersionInfo.versionText);
    PortAudio.Initialize();
    Console.WriteLine($"Number of devices: {PortAudio.DeviceCount}");

    for (int i = 0; i != PortAudio.DeviceCount; ++i)
    {
      Console.WriteLine($" Device {i}");
      DeviceInfo deviceInfo = PortAudio.GetDeviceInfo(i);
      Console.WriteLine($"   Name: {deviceInfo.name}");
      Console.WriteLine($"   Max output channels: {deviceInfo.maxOutputChannels}");
      Console.WriteLine($"   Default sample rate: {deviceInfo.defaultSampleRate}");
    }
    int deviceIndex = PortAudio.DefaultOutputDevice;
    if (deviceIndex == PortAudio.NoDevice)
    {
      Console.WriteLine("No default output device found. Please use ../zipvoice-tts instead");
      Environment.Exit(1);
    }

    var info = PortAudio.GetDeviceInfo(deviceIndex);
    Console.WriteLine();
    Console.WriteLine($"Use output default device {deviceIndex} ({info.name})");

    var param = new StreamParameters();
    param.device = deviceIndex;
    param.channelCount = 1;
    param.sampleFormat = SampleFormat.Float32;
    param.suggestedLatency = info.defaultLowOutputLatency;
    param.hostApiSpecificStreamInfo = IntPtr.Zero;

    var dataItems = new BlockingCollection<float[]>();

    var myCallback = (IntPtr samples, int n, float progress, IntPtr arg) =>
    {
      Console.WriteLine($"Progress {progress * 100}%");

      float[] data = new float[n];
      Marshal.Copy(samples, data, 0, n);
      dataItems.Add(data);

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var playFinished = false;

    float[]? lastSampleArray = null;
    int lastIndex = 0;

    PortAudioSharp.Stream.Callback playCallback = (IntPtr input, IntPtr output,
        UInt32 frameCount,
        ref StreamCallbackTimeInfo timeInfo,
        StreamCallbackFlags statusFlags,
        IntPtr userData
        ) =>
    {
      if (dataItems.IsCompleted && lastSampleArray == null && lastIndex == 0)
      {
        Console.WriteLine("Finished playing");
        playFinished = true;
        return StreamCallbackResult.Complete;
      }

      int expected = Convert.ToInt32(frameCount);
      int i = 0;

      while ((lastSampleArray != null || dataItems.Count != 0) && (i < expected))
      {
        int needed = expected - i;

        if (lastSampleArray != null)
        {
          int remaining = lastSampleArray.Length - lastIndex;
          if (remaining >= needed)
          {
            float[] thisBlock = lastSampleArray.Skip(lastIndex).Take(needed).ToArray();
            lastIndex += needed;
            if (lastIndex == lastSampleArray.Length)
            {
              lastSampleArray = null;
              lastIndex = 0;
            }

            Marshal.Copy(thisBlock, 0, IntPtr.Add(output, i * sizeof(float)), needed);
            return StreamCallbackResult.Continue;
          }

          float[] thisBlock2 = lastSampleArray.Skip(lastIndex).Take(remaining).ToArray();
          lastIndex = 0;
          lastSampleArray = null;

          Marshal.Copy(thisBlock2, 0, IntPtr.Add(output, i * sizeof(float)), remaining);
          i += remaining;
          continue;
        }

        if (dataItems.Count != 0)
        {
          lastSampleArray = dataItems.Take();
          lastIndex = 0;
        }
      }

      if (i < expected)
      {
        int sizeInBytes = (expected - i) * 4;
        Marshal.Copy(new byte[sizeInBytes], 0, IntPtr.Add(output, i * sizeof(float)), sizeInBytes);
      }

      return StreamCallbackResult.Continue;
    };

    PortAudioSharp.Stream stream = new PortAudioSharp.Stream(inParams: null, outParams: param, sampleRate: tts.SampleRate,
        framesPerBuffer: 0,
        streamFlags: StreamFlags.ClipOff,
        callback: playCallback,
        userData: IntPtr.Zero
        );

    stream.Start();

    var callback = new OfflineTtsCallbackProgressWithArg(myCallback);
    var audio = tts.GenerateWithConfig(text, genConfig, callback);

    var outputFilename = "./generated-zipvoice-zh-en-play.wav";
    var ok = audio.SaveToWaveFile(outputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {outputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {outputFilename}");
    }

    dataItems.CompleteAdding();

    while (!playFinished)
    {
      Thread.Sleep(100);
    }
  }
}
