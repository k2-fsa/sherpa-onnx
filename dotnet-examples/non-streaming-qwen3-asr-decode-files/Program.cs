// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a Qwen3 ASR model for speech recognition.
//
// You can find the model doc at
// https://k2-fsa.github.io/sherpa/onnx/qwen3-asr.html
using SherpaOnnx;

class NonStreamingQwen3Asr
{
  static void Main(string[] args)
  {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    var config = new OfflineRecognizerConfig();
    config.ModelConfig.Qwen3Asr.ConvFrontend = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx";
    config.ModelConfig.Qwen3Asr.Encoder = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx";
    config.ModelConfig.Qwen3Asr.Decoder = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx";
    config.ModelConfig.Qwen3Asr.Tokenizer = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer";
    config.ModelConfig.Tokens = "";
    config.ModelConfig.Debug = 1;
    var recognizer = new OfflineRecognizer(config);

    var testWaveFilename = "./sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav";
    var reader = new WaveReader(testWaveFilename);
    var stream = recognizer.CreateStream();
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    recognizer.Decode(stream);
    var text = stream.Result.Text;
    Console.WriteLine("Text: {0}", text);
  }
}
