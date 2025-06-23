// Copyright (c)  2025  Xiaomi Corporation
using SherpaOnnx;

class VersionTestDemo
{
  static void Main(string[] args)
  {
    var version = VersionInfo.Version;
    var gitSha1 = VersionInfo.GitSha1;
    var gitDate = VersionInfo.GitDate;

    Console.WriteLine("sherpa-onnx version: {0}", version);
    Console.WriteLine("sherpa-onnx gitSha1: {0}", gitSha1);
    Console.WriteLine("sherpa-onnx gitDate: {0}", gitDate);
  }
}
