// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a non-streaming zipformer CTC model with an HLG
// decoding graph via sherpa-onnx's .NET API.
//
// The HLG graph is passed to the C API through CtcFstDecoderConfig.
//
// 1. Download the model for testing
//
//  curl -SL -O https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
//  tar xvf sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
//  rm sherpa-onnx-zipformer-ctc-en-2023-10-02.tar.bz2
//
// 2. Now run it
//
// dotnet run -c Release

using SherpaOnnx;

class OfflineZipformerCtcHlgDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineRecognizerConfig();
    config.FeatConfig.SampleRate = 16000;
    config.FeatConfig.FeatureDim = 80;
    config.ModelConfig.ZipformerCtc.Model = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/model.int8.onnx";
    config.ModelConfig.Tokens = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/tokens.txt";
    config.ModelConfig.Provider = "cpu";
    config.ModelConfig.NumThreads = 1;
    config.ModelConfig.Debug = 0;

    // This is what the example exercises: the graph and max_active reach the C
    // API through CtcFstDecoderConfig.
    config.CtcFstDecoderConfig.Graph = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/HLG.fst";
    config.CtcFstDecoderConfig.MaxActive = 3000;

    var recognizer = new OfflineRecognizer(config);

    var filename = "./sherpa-onnx-zipformer-ctc-en-2023-10-02/test_wavs/0.wav";
    var waveReader = new WaveReader(filename);
    var s = recognizer.CreateStream();
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

    recognizer.Decode(s);

    var r = s.Result;
    Console.WriteLine("--------------------");
    Console.WriteLine(filename);
    Console.WriteLine("text: {0}", r.Text);
    Console.WriteLine("tokens: [{0}]", string.Join(", ", r.Tokens));
    Console.WriteLine("--------------------");
  }
}
