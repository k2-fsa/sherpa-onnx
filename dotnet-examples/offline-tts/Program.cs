// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use a non-streaming TTS model for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using CommandLine;
using CommandLine.Text;
using SherpaOnnx;

class OfflineTtsDemo
{
  class Options
  {
    [Option("tts-rule-fsts", Required = false, Default = "", HelpText = "path to rule.fst")]
    public string RuleFsts { get; set; } = string.Empty;

    [Option("tts-rule-fars", Required = false, Default = "", HelpText = "path to rule.far")]
    public string RuleFars { get; set; } = string.Empty;

    [Option("dict-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for jieba.")]
    public string DictDir { get; set; } = string.Empty;

    [Option("data-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for espeak-ng.")]
    public string DataDir { get; set; } = string.Empty;

    [Option("length-scale", Required = false, Default = 1, HelpText = "speech speed. Larger->Slower; Smaller->faster")]
    public float LengthScale { get; set; } = 1;

    [Option("noise-scale", Required = false, Default = 0.667f, HelpText = "noise_scale for VITS or Matcha models")]
    public float NoiseScale { get; set; } = 0.667F;

    [Option("vits-noise-scale-w", Required = false, Default = 0.8F, HelpText = "noise_scale_w for VITS models")]
    public float NoiseScaleW { get; set; } = 0.8F;

    [Option("lexicon", Required = false, Default = "", HelpText = "Path to lexicon.txt")]
    public string Lexicon { get; set; } = string.Empty;

    [Option("tokens", Required = true, Default = "", HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; } = string.Empty;

    [Option("tts-max-num-sentences", Required = false, Default = 1, HelpText = "Maximum number of sentences that we process at a time.")]
    public int MaxNumSentences { get; set; } = 1;

    [Option(Required = false, Default = 0, HelpText = "1 to show debug messages.")]
    public int Debug { get; set; } = 0;

    [Option("vits-model", Required = false, HelpText = "Path to VITS model")]
    public string Model { get; set; } = string.Empty;

    [Option("matcha-acoustic-model", Required = false, HelpText = "Path to the acoustic model of Matcha")]
    public string AcousticModel { get; set; } = "";

    [Option("matcha-vocoder", Required = false, HelpText = "Path to the vocoder model of Matcha")]
    public string Vocoder { get; set; } = "";

    [Option("sid", Required = false, Default = 0, HelpText = "Speaker ID")]
    public int SpeakerId { get; set; } = 0;

    [Option("text", Required = true, HelpText = "Text to synthesize")]
    public string Text { get; set; } = string.Empty;

    [Option("output-filename", Required = true, Default = "./generated.wav", HelpText = "Path to save the generated audio")]
    public string OutputFilename { get; set; } = "./generated.wav";
  }

  static void Main(string[] args)
  {
    var parser = new Parser(with => with.HelpWriter = null);
    var parserResult = parser.ParseArguments<Options>(args);

    parserResult
      .WithParsed<Options>(options => Run(options))
      .WithNotParsed(errs => DisplayHelp(parserResult, errs));
  }

  private static void DisplayHelp<T>(ParserResult<T> result, IEnumerable<Error> errs)
  {
    var usage = @"
# matcha-icefall-zh-baker

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-zh-baker.tar.bz2
tar xvf matcha-icefall-zh-baker.tar.bz2
rm matcha-icefall-zh-baker.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

dotnet run \
  --matcha-acoustic-model=./matcha-icefall-zh-baker/model-steps-3.onnx \
  --matcha-vocoder=./hifigan_v2.onnx \
  --lexicon=./matcha-icefall-zh-baker/lexicon.txt \
  --tokens=./matcha-icefall-zh-baker/tokens.txt \
  --dict-dir=./matcha-icefall-zh-baker/dict \
  --tts-rule-fsts=./matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst \
  --debug=1 \
  --output-filename=./matcha-zh.wav \
  --text='某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。'

# matcha-icefall-en_US-ljspeech

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/matcha-icefall-en_US-ljspeech.tar.bz2
tar xvf matcha-icefall-en_US-ljspeech.tar.bz2
rm matcha-icefall-en_US-ljspeech.tar.bz2

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/hifigan_v2.onnx

dotnet run \
  --matcha-acoustic-model=./matcha-icefall-en_US-ljspeech/model-steps-3.onnx \
  --matcha-vocoder=./hifigan_v2.onnx \
  --tokens=./matcha-icefall-zh-baker/tokens.txt \
  --data-dir=./matcha-icefall-en_US-ljspeech/espeak-ng-data \
  --debug=1 \
  --output-filename=./matcha-zh.wav \
  --text='Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.'

# vits-aishell3

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2

dotnet run \
  --vits-model=./vits-icefall-zh-aishell3/model.onnx \
  --tokens=./vits-icefall-zh-aishell3/tokens.txt \
  --lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
  --tts-rule-fsts=./vits-icefall-zh-aishell3/phone.fst,./vits-icefall-zh-aishell3/date.fst,./vits-icefall-zh-aishell3/number.fst \
  --tts-rule-fars=./vits-icefall-zh-aishell3/rule.far \
  --sid=66 \
  --debug=1 \
  --output-filename=./aishell3-66.wav \
  --text=这是一个语音合成测试

# Piper models

wget -qq https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
tar xf vits-piper-en_US-amy-low.tar.bz2

dotnet run \
  --vits-model=./vits-piper-en_US-amy-low/en_US-amy-low.onnx \
  --tokens=./vits-piper-en_US-amy-low/tokens.txt \
  --data-dir=./vits-piper-en_US-amy-low/espeak-ng-data \
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
    var config = new OfflineTtsConfig();
    config.Model.Vits.Model = options.Model;
    config.Model.Vits.Lexicon = options.Lexicon;
    config.Model.Vits.Tokens = options.Tokens;
    config.Model.Vits.DataDir = options.DataDir;
    config.Model.Vits.DictDir = options.DictDir;
    config.Model.Vits.NoiseScale = options.NoiseScale;
    config.Model.Vits.NoiseScaleW = options.NoiseScaleW;
    config.Model.Vits.LengthScale = options.LengthScale;

    config.Model.Matcha.AcousticModel = options.AcousticModel;
    config.Model.Matcha.Vocoder = options.Vocoder;
    config.Model.Matcha.Lexicon = options.Lexicon;
    config.Model.Matcha.Tokens = options.Tokens;
    config.Model.Matcha.DataDir = options.DataDir;
    config.Model.Matcha.DictDir = options.DictDir;
    config.Model.Matcha.NoiseScale = options.NoiseScale;
    config.Model.Matcha.LengthScale = options.LengthScale;

    config.Model.NumThreads = 1;
    config.Model.Debug = options.Debug;
    config.Model.Provider = "cpu";
    config.RuleFsts = options.RuleFsts;
    config.RuleFars = options.RuleFars;
    config.MaxNumSentences = options.MaxNumSentences;

    var tts = new OfflineTts(config);
    var speed = 1.0f / options.LengthScale;
    var sid = options.SpeakerId;
    var audio = tts.Generate(options.Text, speed, sid);
    var ok = audio.SaveToWaveFile(options.OutputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {options.OutputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to write {options.OutputFilename}");
    }
  }
}
