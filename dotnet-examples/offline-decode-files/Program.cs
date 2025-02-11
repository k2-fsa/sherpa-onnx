// Copyright (c)  2023  Xiaomi Corporation
// Copyright (c)  2023 by manyeyes
//
// This file shows how to use a non-streaming model to decode files
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// to download non-streaming models
using CommandLine;
using CommandLine.Text;
using SherpaOnnx;

class OfflineDecodeFiles
{
  class Options
  {
    [Option("sample-rate", Required = false, Default = 16000, HelpText = "Sample rate of the data used to train the model")]
    public int SampleRate { get; set; } = 16000;

    [Option("feat-dim", Required = false, Default = 80, HelpText = "Dimension of the features used to train the model")]
    public int FeatureDim { get; set; } = 80;

    [Option(Required = false, HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; } = string.Empty;

    [Option(Required = false, Default = "", HelpText = "Path to transducer encoder.onnx. Used only for transducer models")]
    public string Encoder { get; set; } = string.Empty;

    [Option(Required = false, Default = "", HelpText = "Path to transducer decoder.onnx. Used only for transducer models")]
    public string Decoder { get; set; } = string.Empty;

    [Option(Required = false, Default = "", HelpText = "Path to transducer joiner.onnx. Used only for transducer models")]
    public string Joiner { get; set; } = string.Empty;

    [Option("model-type", Required = false, Default = "", HelpText = "model type")]
    public string ModelType { get; set; } = string.Empty;

    [Option("whisper-encoder", Required = false, Default = "", HelpText = "Path to whisper encoder.onnx. Used only for whisper models")]
    public string WhisperEncoder { get; set; } = string.Empty;

    [Option("whisper-decoder", Required = false, Default = "", HelpText = "Path to whisper decoder.onnx. Used only for whisper models")]
    public string WhisperDecoder { get; set; } = string.Empty;

    [Option("whisper-language", Required = false, Default = "", HelpText = "Language of the input file. Can be empty")]
    public string WhisperLanguage { get; set; } = string.Empty;

    [Option("whisper-task", Required = false, Default = "transcribe", HelpText = "transcribe or translate")]
    public string WhisperTask { get; set; } = "transcribe";

    [Option("moonshine-preprocessor", Required = false, Default = "", HelpText = "Path to preprocess.onnx. Used only for Moonshine models")]
    public string MoonshinePreprocessor { get; set; } = string.Empty;

    [Option("moonshine-encoder", Required = false, Default = "", HelpText = "Path to encode.onnx. Used only for Moonshine models")]
    public string MoonshineEncoder { get; set; } = string.Empty;

    [Option("moonshine-uncached-decoder", Required = false, Default = "", HelpText = "Path to uncached_decode.onnx. Used only for Moonshine models")]
    public string MoonshineUncachedDecoder { get; set; } = string.Empty;

    [Option("moonshine-cached-decoder", Required = false, Default = "", HelpText = "Path to cached_decode.onnx. Used only for Moonshine models")]
    public string MoonshineCachedDecoder { get; set; } = string.Empty;

    [Option("tdnn-model", Required = false, Default = "", HelpText = "Path to tdnn yesno model")]
    public string TdnnModel { get; set; } = string.Empty;

    [Option(Required = false, HelpText = "Path to model.onnx. Used only for paraformer models")]
    public string Paraformer { get; set; } = string.Empty;

    [Option("nemo-ctc", Required = false, HelpText = "Path to model.onnx. Used only for NeMo CTC models")]
    public string NeMoCtc { get; set; } = string.Empty;

    [Option("telespeech-ctc", Required = false, HelpText = "Path to model.onnx. Used only for TeleSpeech CTC models")]
    public string TeleSpeechCtc { get; set; } = string.Empty;

    [Option("sense-voice-model", Required = false, HelpText = "Path to model.onnx. Used only for SenseVoice CTC models")]
    public string SenseVoiceModel { get; set; } = string.Empty;

    [Option("sense-voice-use-itn", Required = false, HelpText = "1 to use inverse text normalization for sense voice.")]
    public int SenseVoiceUseItn { get; set; } = 1;

    [Option("num-threads", Required = false, Default = 1, HelpText = "Number of threads for computation")]
    public int NumThreads { get; set; } = 1;

    [Option("decoding-method", Required = false, Default = "greedy_search",
            HelpText = "Valid decoding methods are: greedy_search, modified_beam_search")]
    public string DecodingMethod { get; set; } = "greedy_search";

    [Option("rule-fsts", Required = false, Default = "",
            HelpText = "If not empty, path to rule fst for inverse text normalization")]
    public string RuleFsts { get; set; } = string.Empty;

    [Option("max-active-paths", Required = false, Default = 4,
        HelpText = @"Used only when --decoding--method is modified_beam_search.
It specifies number of active paths to keep during the search")]
    public int MaxActivePaths { get; set; } = 4;

    [Option("hotwords-file", Required = false, Default = "", HelpText = "Path to hotwords.txt")]
    public string HotwordsFile { get; set; } = string.Empty;

    [Option("hotwords-score", Required = false, Default = 1.5F, HelpText = "hotwords score")]
    public float HotwordsScore { get; set; } = 1.5F;

    [Option("files", Required = true, HelpText = "Audio files for decoding")]
    public IEnumerable<string> Files { get; set; } = new string[] { };
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
    var usage = @"
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
  --tokens=./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt \
  --paraformer=./sherpa-onnx-paraformer-zh-2023-09-14/model.onnx \
  --files ./sherpa-onnx-zipformer-en-2023-04-01/test_wavs/0.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/0.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/1.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/2.wav \
  ./sherpa-onnx-paraformer-zh-2023-09-14/test_wavs/8k.wav

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

# Whisper

dotnet run \
  --whisper-encoder=./sherpa-onnx-whisper-tiny.en/tiny.en-encoder.onnx \
  --whisper-decoder=./sherpa-onnx-whisper-tiny.en/tiny.en-decoder.onnx \
  --tokens=./sherpa-onnx-whisper-tiny.en/tiny.en-tokens.txt \
  --files ./sherpa-onnx-whisper-tiny.en/test_wavs/0.wav \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/1.wav \
  ./sherpa-onnx-whisper-tiny.en/test_wavs/8k.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html
to download pre-trained whisper models.

# Tdnn yesno

dotnet run \
  --sample-rate=8000 \
  --feat-dim=23 \
  --tokens=./sherpa-onnx-tdnn-yesno/tokens.txt \
  --tdnn-model=./sherpa-onnx-tdnn-yesno/model-epoch-14-avg-2.onnx \
  --files ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_0_1_0_0_0_1.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_0_1_0.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_0_1_1_1.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_0_1_0_0_1.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_1_0_0_0_1.wav \
  ./sherpa-onnx-tdnn-yesno/test_wavs/0_0_1_1_0_1_1_0.wav

Please refer to
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-ctc/yesno/index.html
to download pre-trained Tdnn models.
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
    config.FeatConfig.SampleRate = options.SampleRate;
    config.FeatConfig.FeatureDim = options.FeatureDim;

    config.ModelConfig.Tokens = options.Tokens;

    if (!string.IsNullOrEmpty(options.Encoder))
    {
      // this is a transducer model
      config.ModelConfig.Transducer.Encoder = options.Encoder;
      config.ModelConfig.Transducer.Decoder = options.Decoder;
      config.ModelConfig.Transducer.Joiner = options.Joiner;
    }
    else if (!string.IsNullOrEmpty(options.Paraformer))
    {
      config.ModelConfig.Paraformer.Model = options.Paraformer;
    }
    else if (!string.IsNullOrEmpty(options.NeMoCtc))
    {
      config.ModelConfig.NeMoCtc.Model = options.NeMoCtc;
    }
    else if (!string.IsNullOrEmpty(options.TeleSpeechCtc))
    {
      config.ModelConfig.TeleSpeechCtc = options.TeleSpeechCtc;
    }
    else if (!string.IsNullOrEmpty(options.WhisperEncoder))
    {
      config.ModelConfig.Whisper.Encoder = options.WhisperEncoder;
      config.ModelConfig.Whisper.Decoder = options.WhisperDecoder;
      config.ModelConfig.Whisper.Language = options.WhisperLanguage;
      config.ModelConfig.Whisper.Task = options.WhisperTask;
    }
    else if (!string.IsNullOrEmpty(options.TdnnModel))
    {
      config.ModelConfig.Tdnn.Model = options.TdnnModel;
    }
    else if (!string.IsNullOrEmpty(options.SenseVoiceModel))
    {
      config.ModelConfig.SenseVoice.Model = options.SenseVoiceModel;
      config.ModelConfig.SenseVoice.UseInverseTextNormalization = options.SenseVoiceUseItn;
    }
    else if (!string.IsNullOrEmpty(options.MoonshinePreprocessor))
    {
      config.ModelConfig.Moonshine.Preprocessor = options.MoonshinePreprocessor;
      config.ModelConfig.Moonshine.Encoder = options.MoonshineEncoder;
      config.ModelConfig.Moonshine.UncachedDecoder = options.MoonshineUncachedDecoder;
      config.ModelConfig.Moonshine.CachedDecoder = options.MoonshineCachedDecoder;
    }
    else
    {
      Console.WriteLine("Please provide a model");
      return;
    }

    config.ModelConfig.ModelType = options.ModelType;
    config.DecodingMethod = options.DecodingMethod;
    config.MaxActivePaths = options.MaxActivePaths;
    config.HotwordsFile = options.HotwordsFile;
    config.HotwordsScore = options.HotwordsScore;
    config.RuleFsts = options.RuleFsts;

    config.ModelConfig.Debug = 0;

    var recognizer = new OfflineRecognizer(config);

    var files = options.Files.ToArray();

    // We create a separate stream for each file
    var streams = new List<OfflineStream>();
    streams.EnsureCapacity(files.Length);

    for (int i = 0; i != files.Length; ++i)
    {
      var s = recognizer.CreateStream();

      WaveReader waveReader = new WaveReader(files[i]);
      s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);
      streams.Add(s);
    }

    recognizer.Decode(streams);

    // display results
    for (int i = 0; i != files.Length; ++i)
    {
      var r = streams[i].Result;
      Console.WriteLine("--------------------");
      Console.WriteLine(files[i]);
      Console.WriteLine("Text: {0}", r.Text);
      Console.WriteLine("Tokens: [{0}]", string.Join(", ", r.Tokens));
      if (r.Timestamps != null && r.Timestamps.Length > 0) {
        Console.Write("Timestamps: [");
        var sep = string.Empty;
        for (int k = 0; k != r.Timestamps.Length; ++k)
        {
          Console.Write("{0}{1}", sep, r.Timestamps[k].ToString("0.00"));
          sep = ", ";
        }
        Console.WriteLine("]");
      }
    }
    Console.WriteLine("--------------------");
  }
}
