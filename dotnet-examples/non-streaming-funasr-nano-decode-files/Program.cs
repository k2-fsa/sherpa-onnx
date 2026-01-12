// Copyright (c)  2026  Xiaomi Corporation
//
// This file shows how to use a FunASR Nano model for speech recognition.
//
// You can find the model doc at
// https://k2-fsa.github.io/sherpa/onnx/funasr-nano.html
using SherpaOnnx;

class NonStreamingFunAsrNano
{
  static void Main(string[] args)
  {
    // please download model files from
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
    var config = new OfflineRecognizerConfig();
    config.ModelConfig.FunAsrNano.EncoderAdaptor = "./sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx";
    config.ModelConfig.FunAsrNano.LLM = "./sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx";
    config.ModelConfig.FunAsrNano.Embedding = "./sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx";
    config.ModelConfig.FunAsrNano.Tokenizer = "./sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B";
    config.ModelConfig.Tokens = "";
    config.ModelConfig.Debug = 1;
    var recognizer = new OfflineRecognizer(config);

    var testWaveFilename = "./sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav";
    var reader = new WaveReader(testWaveFilename);
    var stream = recognizer.CreateStream();
    stream.AcceptWaveform(reader.SampleRate, reader.Samples);
    recognizer.Decode(stream);
    var text = stream.Result.Text;
    Console.WriteLine("Text: {0}", text);
  }
}



