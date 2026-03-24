// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to add punctuations to text incrementally.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2
//
// 2. Now run it
//
// dotnet run

using SherpaOnnx;
using System;

class OnlinePunctuationDemo
{
  static void Main(string[] args)
  {
    var config = new OnlinePunctuationConfig();
    config.Model.CnnBiLstm = "./sherpa-onnx-online-punct-en-2024-08-06/model.onnx";
    config.Model.BpeVocab = "./sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab";
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;

    var punct = new OnlinePunctuation(config);

    var textList = new string[] {
        "how are you i am fine thank you",
        "The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry",
    };

    Console.WriteLine("---------");
    foreach (var text in textList)
    {
      string textWithPunct = punct.AddPunct(text);
      Console.WriteLine("Input text: {0}", text);
      Console.WriteLine("Output text: {0}", textWithPunct);
      Console.WriteLine("---------");
    }
  }
}
