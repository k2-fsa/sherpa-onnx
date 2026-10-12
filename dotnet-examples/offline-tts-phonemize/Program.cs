// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming TTS model for text-to-speech
// with the piper-phonemize text frontend.
//
// The text is phonemized outside of sherpa-onnx with the
// com.github.csukuangfj.piper.phonemize NuGet package and the resulting
// phoneme codepoints are passed via OfflineTtsGenerationConfig.PhonemeCodepoints.
// No lexicon file is needed.
//
// The C# counterpart of
// https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-piper-phonemize.py
// and test-offline-tts-matcha-en-phonemize.py
using CommandLine;
using CommandLine.Text;
using PiperPhonemize;
using SherpaOnnx;

class OfflineTtsPhonemizeDemo
{
  class Options
  {
    [Option("length-scale", Required = false, Default = 1, HelpText = "speech speed. Larger->Slower; Smaller->faster")]
    public float LengthScale { get; set; } = 1;

    [Option("noise-scale", Required = false, Default = 0.667f, HelpText = "noise_scale for VITS or Matcha models")]
    public float NoiseScale { get; set; } = 0.667F;

    [Option("vits-noise-scale-w", Required = false, Default = 0.8F, HelpText = "noise_scale_w for VITS models")]
    public float NoiseScaleW { get; set; } = 0.8F;

    [Option("tokens", Required = true, Default = "", HelpText = "Path to tokens.txt")]
    public string Tokens { get; set; } = string.Empty;

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

    [Option("lang", Required = false, Default = "",
        HelpText = "espeak-ng voice for the phonemizer, e.g., en-us. Empty means the language of the model")]
    public string Lang { get; set; } = string.Empty;

    [Option("output-filename", Required = false, Default = "./generated-dotnet-phonemize.wav",
        HelpText = "Path to save the generated audio")]
    public string OutputFilename { get; set; } = "./generated-dotnet-phonemize.wav";
  }

  static void Main(string[] args)
  {
    var parser = new Parser(with =>
        with.HelpWriter = null);
    var parserResult = parser.ParseArguments<Options>(args);
    parserResult.WithParsed(Run)
        .WithNotParsed(errs => Console.WriteLine(HelpText.AutoBuild(parserResult, h => h, e => e)));
  }

  private static void Run(Options options)
  {
    Console.WriteLine($"piper-phonemize version: {PiperPhonemizeApi.GetVersionStr()}");

    string dataDir = FindEspeakNgData();
    if (dataDir == null)
    {
      Console.Error.WriteLine("ERROR: espeak-ng-data not found.");
      Environment.Exit(1);
    }
    Console.WriteLine($"espeak-ng-data: {dataDir}");

    // Note: espeak-ng-data is only used by the piper-phonemize phonemizer.
    // sherpa-onnx itself does not use espeak-ng anymore.
    int sampleRate = PiperPhonemizeApi.Initialize(dataDir);
    if (sampleRate < 0)
    {
      Console.Error.WriteLine("Failed to initialize piper-phonemize");
      Environment.Exit(1);
    }
    Console.WriteLine($"Sample rate: {sampleRate}");

    var config = new OfflineTtsConfig();
    config.Model.Vits.Model = options.Model;
    config.Model.Vits.Tokens = options.Tokens;
    config.Model.Vits.NoiseScale = options.NoiseScale;
    config.Model.Vits.NoiseScaleW = options.NoiseScaleW;
    config.Model.Vits.LengthScale = options.LengthScale;

    config.Model.Matcha.AcousticModel = options.AcousticModel;
    config.Model.Matcha.Vocoder = options.Vocoder;
    config.Model.Matcha.Tokens = options.Tokens;
    config.Model.Matcha.NoiseScale = options.NoiseScale;
    config.Model.Matcha.LengthScale = options.LengthScale;

    config.Model.NumThreads = 1;
    config.Model.Debug = options.Debug;
    config.Model.Provider = "cpu";

    var tts = new OfflineTts(config);
    var speed = 1.0f / options.LengthScale;

    // The espeak-ng voice for the phonemizer is taken from the model metadata.
    // Fall back to en-us if the model has no language information.
    string lang = options.Lang;
    if (string.IsNullOrEmpty(lang))
    {
      lang = tts.Lang;
      if (string.IsNullOrEmpty(lang))
      {
        Console.Error.WriteLine("Warning: the model has no language information. Falling back to en-us.");
        lang = "en-us";
      }
    }
    Console.WriteLine($"Phonemizer language: {lang}");

    int[][] phonemeCodepoints;
    using (var result = PiperPhonemizeApi.Text(options.Text, lang))
    {
      if (result == null)
      {
        Console.Error.WriteLine("Phonemization failed.");
        Environment.Exit(1);
      }

      phonemeCodepoints = new int[result.NumSentences][];
      for (int i = 0; i < result.NumSentences; i++)
      {
        string ipa = result.GetPhonemesAsString(i) ?? "";
        int[] cps = new int[ipa.Length];
        for (int j = 0; j < ipa.Length; j++)
          cps[j] = ipa[j];
        phonemeCodepoints[i] = cps;
      }
    }
    Console.WriteLine($"Phonemized sentences: {phonemeCodepoints.Length}");

    OfflineTtsGenerationConfig genConfig = new OfflineTtsGenerationConfig();
    genConfig.Sid = options.SpeakerId;
    genConfig.Speed = speed;
    genConfig.PhonemeCodepoints = phonemeCodepoints;

    // The text has already been phonemized; pass an empty string here.
    var audio = tts.GenerateWithConfig("", genConfig, null);

    var ok = audio.SaveToWaveFile(options.OutputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {options.OutputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to save {options.OutputFilename}");
    }
  }

    // The piper-phonemize NuGet package copies espeak-ng-data next to the
    // executable (see its .targets file). espeak_Initialize requires a real
    // directory: an empty path falls back to the system default
    // (/usr/share/espeak-ng-data), which may not exist.
    static string FindEspeakNgData()
    {
        string exeDir = AppDomain.CurrentDomain.BaseDirectory;

        string[] candidates = new[]
        {
            Path.Combine(exeDir, "espeak-ng-data"),
            Path.Combine(exeDir, "..", "espeak-ng-data"),
            Path.Combine(exeDir, "..", "..", "..", "espeak-ng-data"),
            "espeak-ng-data",
        };

        foreach (var candidate in candidates)
        {
            if (Directory.Exists(candidate) && File.Exists(Path.Combine(candidate, "phontab")))
                return Path.GetFullPath(candidate);
        }

        return null;
    }
}
