// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use Kokoro TTS (English) with the piper-phonemize
// text frontend. The text is phonemized outside of sherpa-onnx with the
// com.github.csukuangfj.piper.phonemize NuGet package and the resulting
// phoneme codepoints are passed via OfflineTtsGenerationConfig.PhonemeCodepoints.
// No lexicon file is needed.
//
// The C# counterpart of
// https://github.com/k2-fsa/sherpa-onnx/blob/master/python-api-examples/test-offline-tts-kokoro-en-phonemize.py
using System;
using System.Runtime.InteropServices;
using PiperPhonemize;
using SherpaOnnx;

class KokoroTtsPhonemizeDemo
{
  static void Main(string[] args)
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
    config.Model.Kokoro.Model = "./kokoro-en-v0_19/model.onnx";
    config.Model.Kokoro.Voices = "./kokoro-en-v0_19/voices.bin";
    config.Model.Kokoro.Tokens = "./kokoro-en-v0_19/tokens.txt";
    config.Model.Kokoro.LengthScale = 1.0F;

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

    // The espeak-ng voice for the phonemizer is taken from the model metadata.
    string lang = tts.Lang;
    if (string.IsNullOrEmpty(lang))
    {
      Console.Error.WriteLine("Warning: the model has no language information. Falling back to en-us.");
      lang = "en-us";
    }
    Console.WriteLine($"Phonemizer language: {lang}");

    int[][] phonemeCodepoints;
    using (var result = PiperPhonemizeApi.Text(text, lang))
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
    genConfig.Speed = speed;
    genConfig.PhonemeCodepoints = phonemeCodepoints;

    // The text has already been phonemized; pass an empty string here.
    var audio = tts.GenerateWithConfig("", genConfig, null);

    var outputFilename = "./generated-kokoro-en-phonemize.wav";
    var ok = audio.SaveToWaveFile(outputFilename);

    if (ok)
    {
      Console.WriteLine($"Wrote to {outputFilename} succeeded!");
    }
    else
    {
      Console.WriteLine($"Failed to save {outputFilename}");
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
