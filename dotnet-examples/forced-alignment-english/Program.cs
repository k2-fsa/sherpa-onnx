// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Forced alignment of English speech with the sherpa-onnx C# API.
// It aligns the test wav of the english-mfa-v3.1.0 model package with its
// transcript and writes a Praat TextGrid.

using System;
using System.Diagnostics;
using System.IO;
using SherpaOnnx;

class ForcedAlignmentEnglish
{
    static void Main()
    {
        var config = new ForcedAlignerConfig();
        config.ModelDir = "./english-mfa-v3.1.0";
        config.Lexicon = "./english-mfa-v3.1.0/lexicon.txt";

        var beginInit = Stopwatch.GetTimestamp();
        var aligner = new ForcedAligner(config);
        var stream = aligner.CreateStream();
        var endInit = Stopwatch.GetTimestamp();

        var reader = new WaveReader("./english-mfa-v3.1.0/en.wav");
        stream.AcceptWaveform(reader.SampleRate, reader.Samples);

        // The transcript is in English.
        var text = "ask not what your country can do for you, ask what you can do for your country.";

        var begin = Stopwatch.GetTimestamp();
        var result = aligner.Compute(stream, text);
        var end = Stopwatch.GetTimestamp();

        // tiers[0] is always the "phones" tier, tiers[1] the "words" tier
        var phones = result.Tiers[0];
        var words = result.Tiers[1];

        Console.WriteLine($"--- {phones.Name} ({phones.Intervals.Length} intervals) ---");
        foreach (var interval in phones.Intervals)
        {
            Console.WriteLine($"{interval.Start:F2} - {interval.End:F2}  {interval.Text}");
        }

        Console.WriteLine($"--- {words.Name} ({words.Intervals.Length} intervals) ---");
        foreach (var interval in words.Intervals)
        {
            Console.WriteLine($"{interval.Start:F2} - {interval.End:F2}  {interval.Text}");
        }

        File.WriteAllText("en.TextGrid", result.TextGrid);
        Console.WriteLine("wrote en.TextGrid");

        // time statistics, like sherpa-onnx-fa prints
        double initSeconds = (endInit - beginInit) / (double)Stopwatch.Frequency;
        double elapsedSeconds = (end - begin) / (double)Stopwatch.Frequency;
        double audioDuration = reader.Samples.Length / (double)reader.SampleRate;
        Console.WriteLine($"init is done in {initSeconds:F3} s");
        Console.WriteLine($"Elapsed seconds: {elapsedSeconds:F3} s");
        Console.WriteLine(
            $"Real time factor (RTF): {elapsedSeconds:F3} / {audioDuration:F3} " +
            $"= {elapsedSeconds / audioDuration:F3}");

        stream.Dispose();
        aligner.Dispose();
    }
}
