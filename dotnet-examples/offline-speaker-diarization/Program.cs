// Copyright (c)  2024  Xiaomi Corporation
//

// This file shows how to use sherpa-onnx C# API for speaker diarization
/*
Usage:

Step 1: Download a speaker segmentation model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

Step 2: Download a speaker embedding extractor model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

Step 3. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Run it

  dotnet run
*/

using SherpaOnnx;
using System;

class OfflineSpeakerDiarizationDemo
{
  static void Main(string[] args)
  {
    var config = new OfflineSpeakerDiarizationConfig();
    config.Segmentation.Pyannote.Model = "./sherpa-onnx-pyannote-segmentation-3-0/model.onnx";
    config.Embedding.Model = "./3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx";

    // the test wave ./0-four-speakers-zh.wav has 4 speakers, so
    // we set num_clusters to 4
    //
    config.Clustering.NumClusters = 4;
    // If you don't know the number of speakers in the test wave file, please
    // use
    // config.Clustering.Threshold = 0.5; // You need to tune this threshold
    var sd = new OfflineSpeakerDiarization(config);

    var testWaveFile = "./0-four-speakers-zh.wav";
    WaveReader waveReader = new WaveReader(testWaveFile);
    if (sd.SampleRate != waveReader.SampleRate)
    {
      Console.WriteLine($"Expected sample rate: {sd.SampleRate}. Given: {waveReader.SampleRate}");
      return;
    }

    Console.WriteLine("Started");

     // var segments = sd.Process(waveReader.Samples); // this one is also ok

    var MyProgressCallback = (int numProcessedChunks, int numTotalChunks, IntPtr arg) =>
    {
      float progress = 100.0F * numProcessedChunks / numTotalChunks;
      Console.WriteLine("Progress {0}%", String.Format("{0:0.00}", progress));
      return 0;
    };

    var callback = new OfflineSpeakerDiarizationProgressCallback(MyProgressCallback);
    var segments = sd.ProcessWithCallback(waveReader.Samples, callback, IntPtr.Zero);

    foreach (var s in segments)
    {
      Console.WriteLine("{0} -- {1} speaker_{2}", String.Format("{0:0.00}", s.Start), String.Format("{0:0.00}", s.End), s.Speaker);
    }
  }
}
