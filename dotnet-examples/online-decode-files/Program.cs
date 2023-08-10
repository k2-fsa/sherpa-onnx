// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023 by manyeyes
//
// This file shows how to use a streaming model to decode files
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html
// to download streaming models

using CommandLine.Text;
using CommandLine;
using SherpaOnnx;
using System.Collections.Generic;
using System.Linq;
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

    [Option("enable-endpoint", Required = false, Default = false,
        HelpText = "True to enable endpoint detection.")]
    public bool EnableEndpoint { get; set; }

    [Option("rule1-min-trailing-silence", Required = false, Default = 2.4F,
        HelpText = @"An endpoint is detected if trailing silence in seconds is
larger than this value even if nothing has been decoded. Used only when --enable-endpoint is true.")]
    public float Rule1MinTrailingSilence { get; set; }

    [Option("rule2-min-trailing-silence", Required = false, Default = 1.2F,
        HelpText = @"An endpoint is detected if trailing silence in seconds is
larger than this value after something that is not blank has been decoded. Used
only when --enable-endpoint is true.")]
    public float Rule2MinTrailingSilence { get; set; }

    [Option("rule3-min-utterance-length", Required = false, Default = 20.0F,
        HelpText = @"An endpoint is detected if the utterance in seconds is
larger than this value. Used only when --enable-endpoint is true.")]
    public float Rule3MinUtteranceLength { get; set; }

    [Option("files", Required = true, HelpText = "Audio files for decoding")]
    public IEnumerable<string> Files { get; set; }

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
dotnet run \
  --tokens=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt \
  --encoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.onnx \
  --num-threads=2 \
  --decoding-method=modified_beam_search \
  --debug=false \
  --files ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav \
  ./sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/1.wav

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

    string[] files = options.Files.ToArray();

    // We create a separate stream for each file
    List<OnlineStream> streams = new List<OnlineStream>();
    streams.EnsureCapacity(files.Length);

    for (int i = 0; i != files.Length; ++i)
    {
      OnlineStream s = recognizer.CreateStream();

      WaveReader waveReader = new WaveReader(files[i]);
      s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

      float[] tailPadding = new float[(int)(waveReader.SampleRate * 0.3)];
      s.AcceptWaveform(waveReader.SampleRate, tailPadding);
      s.InputFinished();

      streams.Add(s);
    }

    while (true)
    {
      var readyStreams = streams.Where(s => recognizer.IsReady(s));
      if (!readyStreams.Any())
      {
        break;
      }

      recognizer.Decode(readyStreams);
    }

    // display results
    for (int i = 0; i != files.Length; ++i)
    {
      var text = recognizer.GetResult(streams[i]).Text;
      Console.WriteLine("--------------------");
      Console.WriteLine(files[i]);
      Console.WriteLine(text);
    }
    Console.WriteLine("--------------------");
  }
}
