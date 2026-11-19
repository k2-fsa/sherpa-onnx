// Copyright (c)  2026  Xiaomi Corporation (authors: Fangjun Kuang)
//
// Forced alignment of Chinese speech with the sherpa-onnx C# API.
// It aligns the test wav of the zh-aishell3-pinyin model package with its
// transcript and writes a Praat TextGrid.

using System;
using System.Diagnostics;
using System.IO;
using SherpaOnnx;

class ForcedAlignmentChinese
{
    static void Main()
    {
        var config = new ForcedAlignerConfig();
        config.ModelDir = "./zh-aishell3-pinyin";
        config.Lexicon = "./zh-aishell3-pinyin/lexicon.txt";

        var beginInit = Stopwatch.GetTimestamp();
        var aligner = new ForcedAligner(config);
        var stream = aligner.CreateStream();
        var endInit = Stopwatch.GetTimestamp();

        var reader = new WaveReader("./zh-aishell3-pinyin/zh.wav");
        stream.AcceptWaveform(reader.SampleRate, reader.Samples);

        // The transcript contains hanzi.
        var text = "不要问你的国家能为你做什么，而要问你能为你的国家做什么。";

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

        File.WriteAllText("zh.TextGrid", result.TextGrid);
        Console.WriteLine("wrote zh.TextGrid");

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
