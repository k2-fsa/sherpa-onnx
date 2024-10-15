// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to do keyword spotting with sherpa-onnx.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
// tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
//
// 2. Now run it
//
// dotnet run

using SherpaOnnx;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;

using PortAudioSharp;

class KeywordSpotterDemo
{
  static void Main(string[] args)
  {
    var config = new KeywordSpotterConfig();
    config.FeatConfig.SampleRate = 16000;
    config.FeatConfig.FeatureDim = 80;

    config.ModelConfig.Transducer.Encoder = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx";
    config.ModelConfig.Transducer.Decoder = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx";
    config.ModelConfig.Transducer.Joiner = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx";

    config.ModelConfig.Tokens = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt";
    config.ModelConfig.Provider = "cpu";
    config.ModelConfig.NumThreads = 1;
    config.ModelConfig.Debug = 1;
    config.KeywordsFile = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt";

    var kws = new KeywordSpotter(config);

    var filename = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav";

    WaveReader waveReader = new WaveReader(filename);

    Console.WriteLine("----------Use pre-defined keywords----------");

    OnlineStream s = kws.CreateStream();

    Console.WriteLine(PortAudio.VersionInfo.versionText);
    PortAudio.Initialize();

    Console.WriteLine($"Number of devices: {PortAudio.DeviceCount}");
    for (int i = 0; i != PortAudio.DeviceCount; ++i)
    {
      Console.WriteLine($" Device {i}");
      DeviceInfo deviceInfo = PortAudio.GetDeviceInfo(i);
      Console.WriteLine($"   Name: {deviceInfo.name}");
      Console.WriteLine($"   Max input channels: {deviceInfo.maxInputChannels}");
      Console.WriteLine($"   Default sample rate: {deviceInfo.defaultSampleRate}");
    }
    int deviceIndex = PortAudio.DefaultInputDevice;
    if (deviceIndex == PortAudio.NoDevice)
    {
      Console.WriteLine("No default input device found");
      Environment.Exit(1);
    }

    DeviceInfo info = PortAudio.GetDeviceInfo(deviceIndex);

    Console.WriteLine();
    Console.WriteLine($"Use default device {deviceIndex} ({info.name})");

    StreamParameters param = new StreamParameters();
    param.device = deviceIndex;
    param.channelCount = 1;
    param.sampleFormat = SampleFormat.Float32;
    param.suggestedLatency = info.defaultLowInputLatency;
    param.hostApiSpecificStreamInfo = IntPtr.Zero;

    PortAudioSharp.Stream.Callback callback = (IntPtr input, IntPtr output,
        UInt32 frameCount,
        ref StreamCallbackTimeInfo timeInfo,
        StreamCallbackFlags statusFlags,
        IntPtr userData
        ) =>
    {
      float[] samples = new float[frameCount];
      Marshal.Copy(input, samples, 0, (Int32)frameCount);

      s.AcceptWaveform(config.FeatConfig.SampleRate, samples);

      return StreamCallbackResult.Continue;
    };

    PortAudioSharp.Stream stream = new PortAudioSharp.Stream(inParams: param, outParams: null, sampleRate: config.FeatConfig.SampleRate,
        framesPerBuffer: 0,
        streamFlags: StreamFlags.ClipOff,
        callback: callback,
        userData: IntPtr.Zero
        );

    Console.WriteLine(param);
    Console.WriteLine("Started! Please speak");

    stream.Start();

    while (true)
    {
      while (kws.IsReady(s))
      {
        kws.Decode(s);
      }

      var result = kws.GetResult(s);
      if (result.Keyword != "")
      {
        Console.WriteLine("Detected: {0}", result.Keyword);
      }

      Thread.Sleep(200); // ms
    }

    PortAudio.Terminate();
  }
}

