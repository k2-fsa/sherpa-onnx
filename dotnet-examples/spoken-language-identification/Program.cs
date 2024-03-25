// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to do spoken language identification with whisper.
//
// 1. Download a whisper multilingual model. We use a tiny model below.
// Please refer to https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
// to download more models.
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-whisper-tiny.tar.bz2
// tar xvf sherpa-onnx-whisper-tiny.tar.bz2
// rm sherpa-onnx-whisper-tiny.tar.bz2
//
// 2. Now run it
//
// dotnet run

using SherpaOnnx;
using System.Collections.Generic;
using System;

class SpokenLanguageIdentificationDemo
{

  static void Main(string[] args)
  {
    var config = new SpokenLanguageIdentificationConfig();
    config.Whisper.Encoder = "./sherpa-onnx-whisper-tiny/tiny-encoder.int8.onnx";
    config.Whisper.Decoder = "./sherpa-onnx-whisper-tiny/tiny-decoder.int8.onnx";

    var slid = new SpokenLanguageIdentification(config);
    var filename = "./sherpa-onnx-whisper-tiny/test_wavs/0.wav";

    WaveReader waveReader = new WaveReader(filename);

    var s = slid.CreateStream();
    s.AcceptWaveform(waveReader.SampleRate, waveReader.Samples);
    var result = slid.Compute(s);
    Console.WriteLine($"Filename: {filename}");
    Console.WriteLine($"Detected language: {result.Lang}");
  }
}

