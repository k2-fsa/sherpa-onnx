// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to add punctuations to text.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/punctuation-models/sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12.tar.bz2
//
// 3. Now run it
//
// dotnet run

using SherpaOnnx;

class OfflinePunctuationDemo
{
  static void Main(string[] args)
  {
    var config = new OfflinePunctuationConfig();
    config.Model.CtTransformer = "./sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx";
    config.Model.Debug = 1;
    config.Model.NumThreads = 1;
    var punct = new OfflinePunctuation(config);

    var textList = new string[] {
        "这是一个测试你好吗How are you我很好thank you are you ok谢谢你",
        "我们都是木头人不会说话不会动",
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
