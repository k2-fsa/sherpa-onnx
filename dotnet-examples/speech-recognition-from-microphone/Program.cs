// Copyright (c)  2023  Xiaomi Corporation
//
// This file shows how to use a streaming model for real-time speech
// recognition from a microphone.
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html
// to download streaming models

using CommandLine.Text;
using CommandLine;
using PortAudioSharp;
using System.Threading;
using SherpaOnnx;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System;


class OnlineDecodeFiles
{
  class Options
  {
    [Option(Required = true, HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; }

    [Option(Required = false, Default = "cpu", HelpText = "Provider, e.g., cpu, coreml")]
    public string Provider { get; set; }

    [Option(Required = true, HelpText = "Path to encoder.onnx")]
    public string Encoder { get; set; }

    [Option(Required = true, HelpText = "Path to decoder.onnx")]
    public string Decoder { get; set; }

    [Option(Required = true, HelpText = "Path to joiner.onnx")]
    public string Joiner { get; set; }

    [Option("num-threads", Required = false, Default = 1, HelpText = "Number of threads for computation")]
    public int NumThreads { get; set; }

    [Option("decoding-method", Required = false, Default = "greedy_search",
            HelpText = "Valid decoding methods are: greedy_search, modified_beam_search")]
    public string DecodingMethod { get; set; }

    [Option(Required = false, Default = false, HelpText = "True to show model info during loading")]
    public bool Debug { get; set; }

    [Option("sample-rate", Required = false, Default = 16000, HelpText = "Sample rate of the data used to train the model")]
    public int SampleRate { get; set; }

    [Option("max-active-paths", Required = false, Default = 4,
        HelpText = @"Used only when --decoding--method is modified_beam_search.
It specifies number of active paths to keep during the search")]
    public int MaxActivePaths { get; set; }

    [Option("enable-endpoint", Required = false, Default = true,
        HelpText = "True to enable endpoint detection.")]
    public bool EnableEndpoint { get; set; }

    [Option("rule1-min-trailing-silence", Required = false, Default = 2.4F,
        HelpText = @"An endpoint is detected if trailing silence in seconds is
larger than this value even if nothing has been decoded. Used only when --enable-endpoint is true.")]
    public float Rule1MinTrailingSilence { get; set; }

    [Option("rule2-min-trailing-silence", Required = false, Default = 0.8F,
        HelpText = @"An endpoint is detected if trailing silence in seconds is
larger than this value after something that is not blank has been decoded. Used
only when --enable-endpoint is true.")]
    public float Rule2MinTrailingSilence { get; set; }

    [Option("rule3-min-utterance-length", Required = false, Default = 20.0F,
        HelpText = @"An endpoint is detected if the utterance in seconds is
larger than this value. Used only when --enable-endpoint is true.")]
    public float Rule3MinUtteranceLength { get; set; }
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
dotnet run -c Release \
  --tokens ./icefall-asr-zipformer-streaming-wenetspeech-20230615/data/lang_char/tokens.txt \
  --encoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --decoder ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx \
  --joiner ./icefall-asr-zipformer-streaming-wenetspeech-20230615/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx \

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/index.html
to download pre-trained streaming models.
";

    var helpText = HelpText.AutoBuild(result, h =>
    {
      h.AdditionalNewLineAfterOption = false;
      h.Heading = usage;
      h.Copyright = "Copyright (c) 2023 Xiaomi Corporation";
      return HelpText.DefaultParsingErrorsHandler(result, h);
    }, e => e);
    Console.WriteLine(helpText);
  }

  private static void Run(Options options)
  {
    OnlineRecognizerConfig config = new OnlineRecognizerConfig();
    config.FeatConfig.SampleRate = options.SampleRate;

    // All models from icefall using feature dim 80.
    // You can change it if your model has a different feature dim.
    config.FeatConfig.FeatureDim = 80;

    config.TransducerModelConfig.Encoder = options.Encoder;
    config.TransducerModelConfig.Decoder = options.Decoder;
    config.TransducerModelConfig.Joiner = options.Joiner;
    config.TransducerModelConfig.Tokens = options.Tokens;
    config.TransducerModelConfig.Provider = options.Provider;
    config.TransducerModelConfig.NumThreads = options.NumThreads;
    config.TransducerModelConfig.Debug = options.Debug ? 1 : 0;

    config.DecodingMethod = options.DecodingMethod;
    config.MaxActivePaths = options.MaxActivePaths;
    config.EnableEndpoint = options.EnableEndpoint ? 1 : 0;

    config.Rule1MinTrailingSilence = options.Rule1MinTrailingSilence;
    config.Rule2MinTrailingSilence = options.Rule2MinTrailingSilence;
    config.Rule3MinUtteranceLength = options.Rule3MinUtteranceLength;

    OnlineRecognizer recognizer = new OnlineRecognizer(config);


    OnlineStream s = recognizer.CreateStream();

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

      s.AcceptWaveform(options.SampleRate, samples);

      return StreamCallbackResult.Continue;
    };

    PortAudioSharp.Stream stream = new PortAudioSharp.Stream(inParams: param, outParams: null, sampleRate: options.SampleRate,
        framesPerBuffer: 0,
        streamFlags: StreamFlags.ClipOff,
        callback: callback,
        userData: IntPtr.Zero
        );

    Console.WriteLine(param);

    stream.Start();

    int segment_index = 0;
    String lastText = "";
    int segmentIndex = 0;

    while (true)
    {
      while (recognizer.IsReady(s))
      {
        recognizer.Decode(s);
      }

      var text = recognizer.GetResult(s).Text;
      bool isEndpoint = recognizer.IsEndpoint(s);
      if (!string.IsNullOrWhiteSpace(text) && lastText != text)
      {
        lastText = text;
        Console.Write($"\r{segmentIndex}: {lastText}");
      }

      if (isEndpoint)
      {
        if (!string.IsNullOrWhiteSpace(text))
        {
          ++segmentIndex;
          Console.WriteLine();
        }
        recognizer.Reset(s);
      }

      Thread.Sleep(200); // ms
    }

    PortAudio.Terminate();


  }
}
