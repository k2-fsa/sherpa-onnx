// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use a non-streaming TTS model for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
//
// Note that you need a speaker to run this file since it will play
// the generated audio as it is generating.

using CommandLine.Text;
using CommandLine;
using PortAudioSharp;
using SherpaOnnx;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using System;

class OfflineTtsPlayDemo
{
  class Options
  {

    [Option("tts-rule-fsts", Required = false, Default = "", HelpText = "path to rule.fst")]
    public string RuleFsts { get; set; }

    [Option("vits-dict-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for jieba.")]
    public string DictDir { get; set; }

    [Option("vits-data-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for espeak-ng.")]
    public string DataDir { get; set; }

    [Option("vits-length-scale", Required = false, Default = 1, HelpText = "speech speed. Larger->Slower; Smaller->faster")]
    public float LengthScale { get; set; }

    [Option("vits-noise-scale", Required = false, Default = 0.667f, HelpText = "noise_scale for VITS models")]
    public float NoiseScale { get; set; }

    [Option("vits-noise-scale-w", Required = false, Default = 0.8f, HelpText = "noise_scale_w for VITS models")]
    public float NoiseScaleW { get; set; }

    [Option("vits-lexicon", Required = false, Default = "", HelpText = "Path to lexicon.txt")]
    public string Lexicon { get; set; }

    [Option("vits-tokens", Required = false, Default = "", HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; }

    [Option("tts-max-num-sentences", Required = false, Default = 1, HelpText = "Maximum number of sentences that we process at a time.")]
    public int MaxNumSentences { get; set; }

    [Option(Required = false, Default = 0, HelpText = "1 to show debug messages.")]
    public int Debug { get; set; }

    [Option("vits-model", Required = true, HelpText = "Path to VITS model")]
    public string Model { get; set; }

    [Option("sid", Required = false, Default = 0, HelpText = "Speaker ID")]
    public int SpeakerId { get; set; }

    [Option("text", Required = true, HelpText = "Text to synthesize")]
    public string Text { get; set; }

    [Option("output-filename", Required = true, Default = "./generated.wav", HelpText = "Path to save the generated audio")]
    public string OutputFilename { get; set; }
  }

  static void Main(string[] args)
  {
    var parser = new CommandLine.Parser(with => with.HelpWriter = null);
    var parserResult = parser.ParseArguments<Options>(args);

    parserResult
      .WithParsed<Options>(options => Run(options))
      .WithNotParsed(errs => DisplayHelp(parserResult, errs));
  }

  private static void DisplayHelp<T>(ParserResult<T> result, IEnumerable<Error> errs)
  {
    string usage = @"
# vits-aishell3

wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-zh-aishell3.tar.bz2
tar xf vits-zh-aishell3.tar.bz2

dotnet run \
  --vits-model=./vits-zh-aishell3/vits-aishell3.onnx \
  --vits-tokens=./vits-zh-aishell3/tokens.txt \
  --vits-lexicon=./vits-zh-aishell3/lexicon.txt \
  --tts-rule-fsts=./vits-zh-aishell3/rule.fst \
  --sid=66 \
  --debug=1 \
  --output-filename=./aishell3-66.wav \
  --text=这是一个语音合成测试

# Piper models

wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

dotnet run \
  --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
  --vits-tokens=./vits-piper-en_US-amy-low/tokens.txt \
  --vits-data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
  --debug=1 \
  --output-filename=./amy.wav \
  --text='This is a text to speech application in dotnet with Next Generation Kaldi'

Please refer to
https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
to download more models.
";

    var helpText = HelpText.AutoBuild(result, h =>
    {
      h.AdditionalNewLineAfterOption = false;
      h.Heading = usage;
      h.Copyright = "Copyright (c) 2024 Xiaomi Corporation";
      return HelpText.DefaultParsingErrorsHandler(result, h);
    }, e => e);
    Console.WriteLine(helpText);
  }


  private static void Run(Options options)
  {
    OfflineTtsConfig config = new OfflineTtsConfig();
    config.Model.Vits.Model = options.Model;
    config.Model.Vits.Lexicon = options.Lexicon;
    config.Model.Vits.Tokens = options.Tokens;
    config.Model.Vits.DataDir = options.DataDir;
    config.Model.Vits.DictDir = options.DictDir;
    config.Model.Vits.NoiseScale = options.NoiseScale;
    config.Model.Vits.NoiseScaleW = options.NoiseScaleW;
    config.Model.Vits.LengthScale = options.LengthScale;
    config.Model.NumThreads = 1;
    config.Model.Debug = options.Debug;
    config.Model.Provider = "cpu";
    config.RuleFsts = options.RuleFsts;
    config.MaxNumSentences = options.MaxNumSentences;

    OfflineTts tts = new OfflineTts(config);
    float speed = 1.0f / options.LengthScale;
    int sid = options.SpeakerId;


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

    DeviceInfo info = PortAudio.GetDeviceInfo(deviceIndex);
    Console.WriteLine();
    Console.WriteLine($"Use output default device {deviceIndex} ({info.name})");

    StreamParameters param = new StreamParameters();
    param.device = deviceIndex;
    param.channelCount = 1;
    param.sampleFormat = SampleFormat.Float32;
    param.suggestedLatency = info.defaultLowOutputLatency;
    param.hostApiSpecificStreamInfo = IntPtr.Zero;

    // https://learn.microsoft.com/en-us/dotnet/standard/collections/thread-safe/blockingcollection-overview
    BlockingCollection<float[]> dataItems = new BlockingCollection<float[]>();

    var MyCallback = (IntPtr samples, int n) =>
    {
      float[] data = new float[n];

      Marshal.Copy(samples, data, 0, n);

      dataItems.Add(data);
    };

    bool playFinished = false;

    float[] lastSampleArray = null;
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

    OfflineTtsCallback callback = new OfflineTtsCallback(MyCallback);

    OfflineTtsGeneratedAudio audio = tts.GenerateWithCallback(options.Text, speed, sid, callback);
    bool ok = audio.SaveToWaveFile(options.OutputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {options.OutputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {options.OutputFilename}");
    }
    dataItems.CompleteAdding();

    while (!playFinished)
    {
      Thread.Sleep(100); // 100ms
    }
  }
}
