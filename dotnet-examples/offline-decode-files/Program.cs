// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023 by manyeyes
//
// This file shows how to use a non-streaming model to decode files
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// to download non-streaming models
using CommandLine.Text;
using CommandLine;
using SherpaOnnx;
using System.Collections.Generic;
using System;

class OfflineDecodeFiles
{
  class Options
  {
    [Option(Required = false, HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; }

    [Option(Required = false, HelpText = "Path to encoder.onnx. Used only for transducer models")]
    public string Encoder { get; set; }

    [Option(Required = false, HelpText = "Path to decoder.onnx. Used only for transducer models")]
    public string Decoder { get; set; }

    [Option(Required = false, HelpText = "Path to joiner.onnx. Used only for transducer models")]
    public string Joiner { get; set; }

    [Option(Required = false, HelpText = "Path to model.onnx. Used only for paraformer models")]
    public string Paraformer { get; set; }

    [Option("nemo-ctc", Required = false, HelpText = "Path to model.onnx. Used only for NeMo CTC models")]
    public string NeMoCtc { get; set; }

    [Option("num-threads", Required = false, Default = 1, HelpText = "Number of threads for computation")]
    public int NumThreads { get; set; }

    [Option("decoding-method", Required = false, Default = "greedy_search",
            HelpText = "Valid decoding methods are: greedy_search, modified_beam_search")]
    public string DecodingMethod { get; set; }

    [Option("max-active-paths", Required = false, Default = 4,
        HelpText = @"Used only when --decoding--method is modified_beam_search.
It specifies number of active paths to keep during the search")]
    public int MaxActivePaths { get; set; }

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
# Zipformer

dotnet run \
  --tokens=./sherpa-onnx-zipformer-en-2023-04-01/tokens.txt \
  --encoder=./sherpa-onnx-zipformer-en-2023-04-01/encoder-epoch-99-avg-1.onnx \
  --decoder=./sherpa-onnx-zipformer-en-2023-04-01/decoder-epoch-99-avg-1.onnx \
  --joiner=./sherpa-onnx-zipformer-en-2023-04-01/joiner-epoch-99-avg-1.onnx \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/1.wav \
  ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/8k.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/index.html
to download pre-trained non-streaming zipformer models.

# Paraformer

dotnet run \
  --tokens=./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt \
  --paraformer=./sherpa-onnx-paraformer-zh-2023-03-28/model.onnx \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/0.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/1.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/2.wav \
  ./sherpa-onnx-paraformer-zh-2023-03-28/test_wavs/8k.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
to download pre-trained paraformer models

# NeMo CTC

dotnet run \
  --tokens=./sherpa-onnx-nemo-ctc-en-conformer-medium/tokens.txt \
  --nemo-ctc=./sherpa-onnx-nemo-ctc-en-conformer-medium/model.onnx \
  --num-threads=1 \
  --files ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/0.wav \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/1.wav \
  ./sherpa-onnx-nemo-ctc-en-conformer-medium/test_wavs/8k.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/index.html
to download pre-trained paraformer models
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
    OfflineRecognizerConfig config = new OfflineRecognizerConfig();
    config.ModelConfig.Tokens = options.Tokens;

    if (!String.IsNullOrEmpty(options.Encoder))
    {
      // this is a transducer model
      config.ModelConfig.Transducer.Encoder = options.Encoder;
      config.ModelConfig.Transducer.Decoder = options.Decoder;
      config.ModelConfig.Transducer.Joiner = options.Joiner;
    }
    else if (!String.IsNullOrEmpty(options.Paraformer))
    {
      config.ModelConfig.Paraformer.Model = options.Paraformer;
    }
    else if (!String.IsNullOrEmpty(options.NeMoCtc))
    {
      config.ModelConfig.NeMoCtc.Model = options.NeMoCtc;
    }
    else
    {
      Console.WriteLine("Please provide a model");
      return;
    }

    config.DecodingMethod = options.DecodingMethod;
    config.MaxActivePaths = options.MaxActivePaths;
    config.ModelConfig.Debug = 0;

    OfflineRecognizer recognizer = new OfflineRecognizer(config);

    string[] files = options.Files.ToArray();

    // We create a separate stream for each file
    List<OfflineStream> streams = new List<OfflineStream>();
    streams.EnsureCapacity(files.Length);

    for (int i = 0; i != files.Length; ++i)
    {
      OfflineStream s = recognizer.CreateStream();

      WaveReader waveReader = new WaveReader(files[i]);
      s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);
      streams.Add(s);
    }

    recognizer.Decode(streams);

    // display results
    for (int i = 0; i != files.Length; ++i)
    {
      var text = streams[i].Result.Text;
      Console.WriteLine("--------------------");
      Console.WriteLine(files[i]);
      Console.WriteLine(text);
    }
    Console.WriteLine("--------------------");
  }
}
