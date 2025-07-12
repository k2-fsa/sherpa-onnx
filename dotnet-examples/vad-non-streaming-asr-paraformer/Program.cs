// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use a silero_vad model or ten-vad model
// with a non-streaming Paraformer for speech recognition.
using SherpaOnnx;
using System.IO;


class VadNonStreamingAsrParaformer
{
  static void Main(string[] args)
  {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    var config = new OfflineRecognizerConfig();
    config.ModelConfig.Paraformer.Model = "./sherpa-onnx-paraformer-zh-2023-09-14/model.int8.onnx";
    config.ModelConfig.Tokens = "./sherpa-onnx-paraformer-zh-2023-09-14/tokens.txt";
    config.ModelConfig.Debug = 0;
    var recognizer = new OfflineRecognizer(config);

    var vadModelConfig = new VadModelConfig();
    if (File.Exists("./silero_vad.onnx"))
    {
      Console.WriteLine("Use silero-vad");
      vadModelConfig.SileroVad.Model = "./silero_vad.onnx";
      vadModelConfig.SileroVad.Threshold = 0.3F;
      vadModelConfig.SileroVad.MinSilenceDuration = 0.5F;
      vadModelConfig.SileroVad.MinSpeechDuration = 0.25F;
      vadModelConfig.SileroVad.MaxSpeechDuration = 5.0F;
      vadModelConfig.SileroVad.WindowSize = 512;
    }
    else if (File.Exists("./ten-vad.onnx"))
    {
      Console.WriteLine("Use ten-vad");
      vadModelConfig.TenVad.Model = "./ten-vad.onnx";
      vadModelConfig.TenVad.Threshold = 0.3F;
      vadModelConfig.TenVad.MinSilenceDuration = 0.5F;
      vadModelConfig.TenVad.MinSpeechDuration = 0.25F;
      vadModelConfig.TenVad.MaxSpeechDuration = 5.0F;
      vadModelConfig.TenVad.WindowSize = 256;
    }
    else
    {
      Console.WriteLine("Please download ./silero_vad.onnx or ./ten-vad.onnx");
      return;
    }
    vadModelConfig.Debug = 0;

    var vad = new VoiceActivityDetector(vadModelConfig, 60);

    var testWaveFilename = "./lei-jun-test.wav";
    var reader = new WaveReader(testWaveFilename);

    int numSamples = reader.Samples.Length;
    int windowSize = vadModelConfig.SileroVad.WindowSize;

    if (vadModelConfig.TenVad.Model != "")
    {
      windowSize = vadModelConfig.TenVad.WindowSize;
    }

    int sampleRate = vadModelConfig.SampleRate;
    int numIter = numSamples / windowSize;

    for (int i = 0; i != numIter; ++i)
    {
      int start = i * windowSize;
      var samples = new float[windowSize];
      Array.Copy(reader.Samples, start, samples, 0, windowSize);
      vad.AcceptWaveform(samples);
      if (vad.IsSpeechDetected())
      {
        while (!vad.IsEmpty())
        {
          SpeechSegment segment = vad.Front();
          var startTime = segment.Start / (float)sampleRate;
          var duration = segment.Samples.Length / (float)sampleRate;

          OfflineStream stream = recognizer.CreateStream();
          stream.AcceptWaveform(sampleRate, segment.Samples);
          recognizer.Decode(stream);
          var text = stream.Result.Text;

          if (!string.IsNullOrEmpty(text))
          {
            Console.WriteLine("{0}--{1}: {2}", string.Format("{0:0.00}", startTime),
                string.Format("{0:0.00}", startTime + duration), text);
          }

          vad.Pop();
        }
      }
    }

    vad.Flush();

    while (!vad.IsEmpty())
    {
      var segment = vad.Front();
      float startTime = segment.Start / (float)sampleRate;
      float duration = segment.Samples.Length / (float)sampleRate;

      var stream = recognizer.CreateStream();
      stream.AcceptWaveform(sampleRate, segment.Samples);
      recognizer.Decode(stream);
      var text = stream.Result.Text;

      if (!string.IsNullOrEmpty(text))
      {
        Console.WriteLine("{0}--{1}: {2}", string.Format("{0:0.00}", startTime),
            string.Format("{0:0.00}", startTime + duration), text);
      }

      vad.Pop();
    }
  }
}

