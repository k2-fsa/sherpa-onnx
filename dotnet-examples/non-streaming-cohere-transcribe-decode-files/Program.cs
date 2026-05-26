// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a Cohere Transcribe model for speech recognition.
using SherpaOnnx;

class NonStreamingCohereTranscribe
{
  static void Main(string[] args)
  {
    var config = new OfflineRecognizerConfig();
    config.ModelConfig.CohereTranscribe.Encoder = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx";
    config.ModelConfig.CohereTranscribe.Decoder = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx";
    config.ModelConfig.CohereTranscribe.UsePunct = 1;
    config.ModelConfig.CohereTranscribe.UseItn = 1;
    config.ModelConfig.Tokens = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt";
    config.ModelConfig.Debug = 1;
    var recognizer = new OfflineRecognizer(config);

    var testWaveFilename = "./sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav";
    var reader = new WaveReader(testWaveFilename);
    var stream = recognizer.CreateStream();
    stream.SetOption("language", "en");
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    recognizer.Decode(stream);
    var text = stream.Result.Text;
    Console.WriteLine("Text: {0}", text);
  }
}
