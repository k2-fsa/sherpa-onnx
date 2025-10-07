// Copyright (c)  2025  Xiaomi Corporation
//
// This file shows how to use a non-streaming Zipformer or CED model
// for audio tagging
// Please refer to
// https://k2-fsa.github.io/sherpa/onnx/audio-tagging/index.html
// and
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/audio-tagging-models
// to download pre-trained models

using SherpaOnnx;
using System.Runtime.InteropServices;

class AudioTaggingDemo
{
  static void Main(string[] args)
  {
    TestZipformer();
    // TestCED();
  }

  static void TestZipformer()
  {
    var config = new AudioTaggingConfig();

    config.Model.Zipformer.Model =
      "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/model.onnx";

    config.Model.NumThreads = 1;
    config.Model.Debug = 1;
    config.Labels =
        "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/"
        "class_labels_indices.csv";

    config.TopK = 5;

    var tagger = new AudioTagging(config);

    var s = tagger.CreateStream();

    var waveFilename = "./sherpa-onnx-zipformer-small-audio-tagging-2024-04-15/test_wavs/1.wav";
    WaveReader waveReader = new WaveReader(waveFilename);
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);

    var events = tagging.Compute(s);
    foreach (var e in events)
    {
      Console.WriteLine($"Name {e.name}, index: {e.index}, prob: {e.prob}");
    }
  }
}
