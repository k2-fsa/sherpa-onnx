// Copyright (c)  2025  Xiaomi Corporation
//
// This file shows how to use a NeMo Canary model for speech recognition.
//
// You can find the model doc at
// https://k2-fsa.github.io/sherpa/onnx/nemo/canary.html
using SherpaOnnx;

class NonStreamingAsrCanary
{
  static void Main(string[] args)
  {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    var config = new OfflineRecognizerConfig();
    config.ModelConfig.Canary.Encoder = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/encoder.int8.onnx";
    config.ModelConfig.Canary.Decoder = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/decoder.int8.onnx";
    config.ModelConfig.Canary.SrcLang = "en";
    config.ModelConfig.Canary.TgtLang = "en";
    config.ModelConfig.Tokens = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/tokens.txt";
    config.ModelConfig.Debug = 0;
    var recognizer = new OfflineRecognizer(config);

    var testWaveFilename = "./sherpa-onnx-nemo-canary-180m-flash-en-es-de-fr-int8/test_wavs/en.wav";
    var reader = new WaveReader(testWaveFilename);
    var stream = recognizer.CreateStream();
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    recognizer.Decode(stream);
    var text = stream.Result.Text;
    Console.WriteLine("Text (English): {0}", text);

    // Now output text in German
    config.ModelConfig.Canary.TgtLang = "de";
    recognizer.SetConfig(config);

    stream = recognizer.CreateStream();
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    recognizer.Decode(stream);
    text = stream.Result.Text;
    Console.WriteLine("Text (German): {0}", text);
  }
}


