// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a Moonshine v2 model for speech recognition.
//
// You can find the model doc at
// https://k2-fsa.github.io/sherpa/onnx/moonshine/
using SherpaOnnx;

class NonStreamingAsrMoonshineV2
{
  static void Main(string[] args)
  {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    var config = new OfflineRecognizerConfig();
    config.ModelConfig.Moonshine.Encoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/encoder_model.ort";
    config.ModelConfig.Moonshine.MergedDecoder = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/decoder_model_merged.ort";
    config.ModelConfig.Tokens = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/tokens.txt";
    config.ModelConfig.Debug = 0;
    var recognizer = new OfflineRecognizer(config);

    var testWaveFilename = "./sherpa-onnx-moonshine-tiny-en-quantized-2026-02-27/test_wavs/0.wav";
    var reader = new WaveReader(testWaveFilename);
    var stream = recognizer.CreateStream();
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    recognizer.Decode(stream);
    var text = stream.Result.Text;
    Console.WriteLine("Text: {0}", text);
  }
}



