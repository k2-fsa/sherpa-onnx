// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to do speaker identification with sherpa-onnx.
//
// 1. Download a model from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
//
// wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx
//
// 2. Download test data from
//
// git clone https://github.com/csukuangfj/sr-data
//
// 3. Now run it
//
// dotnet run

using SherpaOnnx;

class SpeakerIdentificationDemo
{
  public static float[] ComputeEmbedding(SpeakerEmbeddingExtractor extractor, string filename)
  {
    var reader = new WaveReader(filename);

    var stream = extractor.CreateStream();
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    stream.InputFinished();

    var embedding = extractor.Compute(stream);

    return embedding;
  }

  static void Main(string[] args)
  {
    var config = new SpeakerEmbeddingExtractorConfig();
    config.Model = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";
    config.Debug = 1;
    var extractor = new SpeakerEmbeddingExtractor(config);

    var manager = new SpeakerEmbeddingManager(extractor.Dim);

    var spk1Files =
        new string[] {
          "./sr-data/enroll/fangjun-sr-1.wav",
          "./sr-data/enroll/fangjun-sr-2.wav",
          "./sr-data/enroll/fangjun-sr-3.wav",
        };
    var spk1Vec = new float[spk1Files.Length][];

    for (int i = 0; i < spk1Files.Length; ++i)
    {
      spk1Vec[i] = ComputeEmbedding(extractor, spk1Files[i]);
    }

    var spk2Files =
        new string[] {
          "./sr-data/enroll/leijun-sr-1.wav", "./sr-data/enroll/leijun-sr-2.wav",
        };

    var spk2Vec = new float[spk2Files.Length][];

    for (int i = 0; i < spk2Files.Length; ++i)
    {
      spk2Vec[i] = ComputeEmbedding(extractor, spk2Files[i]);
    }

    if (!manager.Add("fangjun", spk1Vec))
    {
      Console.WriteLine("Failed to register fangjun");
      return;
    }

    if (!manager.Add("leijun", spk2Vec))
    {
      Console.WriteLine("Failed to register leijun");
      return;
    }

    if (manager.NumSpeakers != 2)
    {
      Console.WriteLine("There should be two speakers");
      return;
    }

    if (!manager.Contains("fangjun"))
    {
      Console.WriteLine("It should contain the speaker fangjun");
      return;
    }

    if (!manager.Contains("leijun"))
    {
      Console.WriteLine("It should contain the speaker leijun");
      return;
    }

    Console.WriteLine("---All speakers---");

    var allSpeakers = manager.GetAllSpeakers();
    foreach (var s in allSpeakers)
    {
      Console.WriteLine(s);
    }
    Console.WriteLine("------------");

    var testFiles =
        new string[] {
          "./sr-data/test/fangjun-test-sr-1.wav",
          "./sr-data/test/leijun-test-sr-1.wav",
          "./sr-data/test/liudehua-test-sr-1.wav"
        };

    float threshold = 0.6f;
    foreach (var file in testFiles)
    {
      var embedding = ComputeEmbedding(extractor, file);

      var name = manager.Search(embedding, threshold);
      if (name == "")
      {
        name = "<Unknown>";
      }
      Console.WriteLine("{0}: {1}", file, name);
    }

    // test verify
    if (!manager.Verify("fangjun", ComputeEmbedding(extractor, testFiles[0]), threshold))
    {
      Console.WriteLine("testFiles[0] should match fangjun!");
      return;
    }

    if (!manager.Remove("fangjun"))
    {
      Console.WriteLine("Failed to remove fangjun");
      return;
    }

    if (manager.Verify("fangjun", ComputeEmbedding(extractor, testFiles[0]), threshold))
    {
      Console.WriteLine("{0} should match no one!", testFiles[0]);
      return;
    }

    if (manager.NumSpeakers != 1)
    {
      Console.WriteLine("There should only 1 speaker left.");
      return;
    }
  }
}
