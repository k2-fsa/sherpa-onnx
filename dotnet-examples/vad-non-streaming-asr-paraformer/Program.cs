// Copyright (c)  2024  Xiaomi Corporation
//
// This file shows how to use a silero_vad model with a non-streaming Paraformer
// for speech recognition.
using SherpaOnnx;
using System.Collections.Generic;
using System;

class VadNonStreamingAsrParaformer
{
  static void Main(string[] args)
  {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    OfflineRecognizerConfig config = new OfflineRecognizerConfig();
    config.ModelConfig.Paraformer.Model = "./sherpa-onnx-paraformer-zh-2023-03-28/model.int8.onnx";
    config.ModelConfig.Tokens = "./sherpa-onnx-paraformer-zh-2023-03-28/tokens.txt";
    config.ModelConfig.Debug = 0;
    OfflineRecognizer recognizer = new OfflineRecognizer(config);

    VadModelConfig vadModelConfig = new VadModelConfig();
    vadModelConfig.SileroVad.Model = "./silero_vad.onnx";
    vadModelConfig.Debug = 0;

    VoiceActivityDetector vad = new VoiceActivityDetector(vadModelConfig, 60);

    string testWaveFilename = "./lei-jun-test.wav";
    WaveReader reader = new WaveReader(testWaveFilename);

    int numSamples = reader.Samples.Length;
    int windowSize = vadModelConfig.SileroVad.WindowSize;
    int sampleRate = vadModelConfig.SampleRate;
    int numIter = numSamples / windowSize;

    for (int i = 0; i != numIter; ++i) {
      int start = i * windowSize;
      float[] samples = new float[windowSize];
      Array.Copy(reader.Samples, start, samples, 0, windowSize);
      vad.AcceptWaveform(samples);
      if (vad.IsSpeechDetected()) {
        while (!vad.IsEmpty()) {
          SpeechSegment segment = vad.Front();
          float startTime = segment.Start / (float)sampleRate;
          float duration = segment.Samples.Length / (float)sampleRate;

          OfflineStream stream = recognizer.CreateStream();
          stream.AcceptWaveform(sampleRate, segment.Samples);
          recognizer.Decode(stream);
          String text = stream.Result.Text;

          if (!String.IsNullOrEmpty(text)) {
            Console.WriteLine("{0}--{1}: {2}", String.Format("{0:0.00}", startTime),
                String.Format("{0:0.00}", startTime+duration), text);
          }

          vad.Pop();
        }
      }
    }

    vad.Flush();

    while (!vad.IsEmpty()) {
      SpeechSegment segment = vad.Front();
      float startTime = segment.Start / (float)sampleRate;
      float duration = segment.Samples.Length / (float)sampleRate;

      OfflineStream stream = recognizer.CreateStream();
      stream.AcceptWaveform(sampleRate, segment.Samples);
      recognizer.Decode(stream);
      String text = stream.Result.Text;

      if (!String.IsNullOrEmpty(text)) {
        Console.WriteLine("{0}--{1}: {2}", String.Format("{0:0.00}", startTime),
            String.Format("{0:0.00}", startTime+duration), text);
      }

      vad.Pop();
    }
  }
}

