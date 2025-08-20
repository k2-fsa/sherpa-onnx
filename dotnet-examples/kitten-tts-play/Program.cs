// Copyright (c)  2025  Xiaomi Corporation
//
// This file shows how to use a non-streaming Kitten TTS model
// for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using PortAudioSharp;
using SherpaOnnx;
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

class KittenTtsPlayDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineTtsConfig();
    config.Model.Kitten.Model = "./kitten-nano-en-v0_1-fp16/model.fp16.onnx";
    config.Model.Kitten.Voices = "./kitten-nano-en-v0_1-fp16/voices.bin";
    config.Model.Kitten.Tokens = "./kitten-nano-en-v0_1-fp16/tokens.txt";
    config.Model.Kitten.DataDir = "./kitten-nano-en-v0_1-fp16/espeak-ng-data";

    config.Model.NumThreads = 2;
    config.Model.Debug = 1;
    config.Model.Provider = "cpu";

    var tts = new OfflineTts(config);
    var speed = 1.0f;
    var text = "Today as always, men fall into two groups: slaves and free men. Whoever " +
      "does not have two-thirds of his day for himself, is a slave, whatever " +
      "he may be: a statesman, a businessman, an official, or a scholar. " +
      "Friends fell out often because life was changing so fast. The easiest " +
      "thing in the world was to lose touch with someone.";

    // mapping of sid to voice name
    // 0->expr-voice-2-m, 1->expr-voice-2-f, 2->expr-voice-3-m
    // 3->expr-voice-3-f, 4->expr-voice-4-m, 5->expr-voice-4-f
    // 6->expr-voice-5-m, 7->expr-voice-5-f
    var sid = 0;


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
      Console.WriteLine("No default output device found. Please use ../offline-tts instead");
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

    // https://learn.microsoft.com/en-us/dotnet/standard/collections/thread-safe/blockingcollection-overview
    var dataItems = new BlockingCollection<float[]>();

    var MyCallback = (IntPtr samples, int n, float progress) =>
    {
      Console.WriteLine($"Progress {progress*100}%");

      float[] data = new float[n];

      Marshal.Copy(samples, data, 0, n);

      dataItems.Add(data);

      // 1 means to keep generating
      // 0 means to stop generating
      return 1;
    };

    var playFinished = false;

    float[]? lastSampleArray = null;
    int lastIndex = 0; // not played

    PortAudioSharp.Stream.Callback playCallback = (IntPtr input, IntPtr output,
        UInt32 frameCount,
        ref StreamCallbackTimeInfo timeInfo,
        StreamCallbackFlags statusFlags,
        IntPtr userData
        ) =>
    {
      if (dataItems.IsCompleted && lastSampleArray == null && lastIndex == 0)
      {
        Console.WriteLine($"Finished playing");
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
            float[] this_block = lastSampleArray.Skip(lastIndex).Take(needed).ToArray();
            lastIndex += needed;
            if (lastIndex == lastSampleArray.Length)
            {
              lastSampleArray = null;
              lastIndex = 0;
            }

            Marshal.Copy(this_block, 0, IntPtr.Add(output, i * sizeof(float)), needed);
            return StreamCallbackResult.Continue;
          }

          float[] this_block2 = lastSampleArray.Skip(lastIndex).Take(remaining).ToArray();
          lastIndex = 0;
          lastSampleArray = null;

          Marshal.Copy(this_block2, 0, IntPtr.Add(output, i * sizeof(float)), remaining);
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

    var callback = new OfflineTtsCallbackProgress(MyCallback);

    var audio = tts.GenerateWithCallbackProgress(text, speed, sid, callback);
    var outputFilename = "./generated-kitten-0.wav";
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
      Thread.Sleep(100); // 100ms
    }
  }
}
