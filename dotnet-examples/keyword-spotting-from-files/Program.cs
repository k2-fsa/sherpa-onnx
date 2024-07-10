// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to do keyword spotting with sherpa-onnx.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/kws-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
// tar xvf sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
//
// 2. Now run it
//
// dotnet run

using SherpaOnnx;
using System.Collections.Generic;
using System;

class KeywordSpotterDemo
{
  static void Main(string[] args)
  {
    var config = new KeywordSpotterConfig();
    config.FeatConfig.SampleRate = 16000;
    config.FeatConfig.FeatureDim = 80;

    config.ModelConfig.Transducer.Encoder = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/encoder-epoch-12-avg-2-chunk-16-left-64.onnx";
    config.ModelConfig.Transducer.Decoder = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/decoder-epoch-12-avg-2-chunk-16-left-64.onnx";
    config.ModelConfig.Transducer.Joiner = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/joiner-epoch-12-avg-2-chunk-16-left-64.onnx";

    config.ModelConfig.Tokens = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/tokens.txt";
    config.ModelConfig.Provider = "cpu";
    config.ModelConfig.NumThreads = 1;
    config.ModelConfig.Debug = 1;
    config.KeywordsFile = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/test_keywords.txt";

    var kws = new KeywordSpotter(config);

    var filename = "./sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav";

    WaveReader waveReader = new WaveReader(filename);

    Console.WriteLine("----------Use pre-defined keywords----------");

    OnlineStream s = kws.CreateStream();
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

    float[] tailPadding = new float[(int)(waveReader.SampleRate * 0.3)];
    s.AcceptWaveform(waveReader.SampleRate, tailPadding);
    s.InputFinished();

    while (kws.IsReady(s))
    {
      kws.Decode(s);
      var result = kws.GetResult(s);
      if (result.Keyword != "")
      {
        Console.WriteLine("Detected: {0}", result.Keyword);
      }
    }

    Console.WriteLine("----------Use pre-defined keywords + add a new keyword----------");
    s = kws.CreateStream("y ǎn y uán @演员");
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

    s.AcceptWaveform(waveReader.SampleRate, tailPadding);
    s.InputFinished();

    while (kws.IsReady(s))
    {
      kws.Decode(s);
      var result = kws.GetResult(s);
      if (result.Keyword != "")
      {
        Console.WriteLine("Detected: {0}", result.Keyword);
      }
    }

    Console.WriteLine("----------Use pre-defined keywords + add 2 new keywords----------");

    // Note keywords are separated by /
    s = kws.CreateStream("y ǎn y uán @演员/zh ī m íng @知名");
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

    s.AcceptWaveform(waveReader.SampleRate, tailPadding);
    s.InputFinished();

    while (kws.IsReady(s))
    {
      kws.Decode(s);
      var result = kws.GetResult(s);
      if (result.Keyword != "")
      {
        Console.WriteLine("Detected: {0}", result.Keyword);
      }
    }
  }
}

