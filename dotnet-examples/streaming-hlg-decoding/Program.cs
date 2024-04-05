// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to do streaming HLG decoding.
//
// 1. Download the model for testing
//
//  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
//  tar xvf sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
//  rm sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18.tar.bz2
//
// 2. Now run it
//
// dotnet run

using SherpaOnnx;
using System.Collections.Generic;
using System;

class StreamingHlgDecodingDemo
{

  static void Main(string[] args)
  {
    var config = new OnlineRecognizerConfig();
    config.FeatConfig.SampleRate = 16000;
    config.FeatConfig.FeatureDim = 80;
    config.ModelConfig.Zipformer2Ctc.Model = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx";

    config.ModelConfig.Tokens = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt";
    config.ModelConfig.Provider = "cpu";
    config.ModelConfig.NumThreads = 1;
    config.ModelConfig.Debug = 0;
    config.CtcFstDecoderConfig.Graph = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst";

    OnlineRecognizer recognizer = new OnlineRecognizer(config);

    var filename = "./sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav";

    WaveReader waveReader = new WaveReader(filename);
    OnlineStream s = recognizer.CreateStream();
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

    float[] tailPadding = new float[(int)(waveReader.SampleRate * 0.3)];
    s.AcceptWaveform(waveReader.SampleRate, tailPadding);
    s.InputFinished();

    while (recognizer.IsReady(s))
    {
      recognizer.Decode(s);
    }

    OnlineRecognizerResult r = recognizer.GetResult(s);
    var text = r.Text;
    var tokens = r.Tokens;
    Console.WriteLine("--------------------");
    Console.WriteLine(filename);
    Console.WriteLine("text: {0}", text);
    Console.WriteLine("tokens: [{0}]", string.Join(", ", tokens));
    Console.Write("timestamps: [");
    r.Timestamps.ToList().ForEach(i => Console.Write(String.Format("{0:0.00}", i) + ", "));
    Console.WriteLine("]");
    Console.WriteLine("--------------------");
  }
}


