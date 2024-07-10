// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use a non-streaming TTS model for text-to-speech
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
// to download pre-trained models
using CommandLine.Text;
using CommandLine;
using SherpaOnnx;
using System.Collections.Generic;
using System;

class OfflineTtsDemo
{
  class Options
  {

    [Option("tts-rule-fsts", Required = false, Default = "", HelpText = "path to rule.fst")]
    public string RuleFsts { get; set; } = "";

    [Option("tts-rule-fars", Required = false, Default = "", HelpText = "path to rule.far")]
    public string RuleFars { get; set; } = "";

    [Option("vits-dict-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for jieba.")]
    public string DictDir { get; set; } = "";

    [Option("vits-data-dir", Required = false, Default = "", HelpText = "Path to the directory containing dict for espeak-ng.")]
    public string DataDir { get; set; } = "";

    [Option("vits-length-scale", Required = false, Default = 1, HelpText = "speech speed. Larger->Slower; Smaller->faster")]
    public float LengthScale { get; set; } = 1;

    [Option("vits-noise-scale", Required = false, Default = 0.667f, HelpText = "noise_scale for VITS models")]
    public float NoiseScale { get; set; } = 0.667F;

    [Option("vits-noise-scale-w", Required = false, Default = 0.8F, HelpText = "noise_scale_w for VITS models")]
    public float NoiseScaleW { get; set; } = 0.8F;

    [Option("vits-lexicon", Required = false, Default = "", HelpText = "Path to lexicon.txt")]
    public string Lexicon { get; set; } = "";

    [Option("vits-tokens", Required = false, Default = "", HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; } = "";

    [Option("tts-max-num-sentences", Required = false, Default = 1, HelpText = "Maximum number of sentences that we process at a time.")]
    public int MaxNumSentences { get; set; } = 1;

    [Option(Required = false, Default = 0, HelpText = "1 to show debug messages.")]
    public int Debug { get; set; } = 0;

    [Option("vits-model", Required = true, HelpText = "Path to VITS model")]
    public string Model { get; set; } = "";

    [Option("sid", Required = false, Default = 0, HelpText = "Speaker ID")]
    public int SpeakerId { get; set; } = 0;

    [Option("text", Required = true, HelpText = "Text to synthesize")]
    public string Text { get; set; } = "";

    [Option("output-filename", Required = true, Default = "./generated.wav", HelpText = "Path to save the generated audio")]
    public string OutputFilename { get; set; } = "./generated.wav";
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

curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
tar xvf vits-icefall-zh-aishell3.tar.bz2

dotnet run \
  --vits-model=./vits-icefall-zh-aishell3/model.onnx \
  --vits-tokens=./vits-icefall-zh-aishell3/tokens.txt \
  --vits-lexicon=./vits-icefall-zh-aishell3/lexicon.txt \
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
    config.RuleFars = options.RuleFars;
    config.MaxNumSentences = options.MaxNumSentences;

    OfflineTts tts = new OfflineTts(config);
    float speed = 1.0f / options.LengthScale;
    int sid = options.SpeakerId;
    OfflineTtsGeneratedAudio audio = tts.Generate(options.Text, speed, sid);
    bool ok = audio.SaveToWaveFile(options.OutputFilename);

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
