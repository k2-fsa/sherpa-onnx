{ Copyright (c)  2024  Xiaomi Corporation }

{
This file shows how to use a streaming Zipformer transducer
to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program zipformer_transducer;

{$mode objfpc}

uses
  sherpa_onnx,
  SysUtils;

var
  Config: TSherpaOnnxOnlineRecognizerConfig;
  Recognizer: TSherpaOnnxOnlineRecognizer;
  Stream: TSherpaOnnxOnlineStream;
  RecognitionResult: TSherpaOnnxOnlineRecognizerResult;
  Wave: TSherpaOnnxWave;
  WaveFilename: AnsiString;
  TailPaddings: array of Single;
begin
  Initialize(Config);

  {Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  to download model files used in this file.}
  Config.ModelConfig.Paraformer.Encoder := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx';
  Config.ModelConfig.Paraformer.Decoder := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx';
  Config.ModelConfig.Tokens := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt"';
  Config.ModelConfig.Debug := False;

  WaveFilename := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/2.wav';

  Recognizer := TSherpaOnnxOnlineRecognizer.Create(Config);

  Stream := Recognizer.CreateStream();

  Wave := SherpaOnnxReadWave(WaveFilename);

  Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);

  SetLength(TailPaddings, Round(Wave.SampleRate * 0.5)); {0.5 seconds of padding}
  Stream.AcceptWaveform(TailPaddings, Wave.SampleRate);

  Stream.InputFinished();

  while Recognizer.IsReady(Stream) do
    Recognizer.Decode(Stream);

  RecognitionResult := Recognizer.GetResult(Stream);
  WriteLn(RecognitionResult.ToString);


  {Free resources to avoid memory leak.

  Note: You don't need to invoke them for this simple script.
  However, you have to invoke them in your own large/complex project.
  }
  FreeAndNil(Stream);
  FreeAndNil(Recognizer);
end.
