{ Copyright (c)  2024  Xiaomi Corporation }
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
begin
  Initialize(Config);

  {Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  to download model files used in this file.}
  Config.ModelConfig.Transducer.Encoder := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/encoder-epoch-99-avg-1.int8.onnx';
  Config.ModelConfig.Transducer.Decoder := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/decoder-epoch-99-avg-1.onnx';
  Config.ModelConfig.Transducer.Joiner := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/joiner-epoch-99-avg-1.int8.onnx';
  Config.ModelConfig.Tokens := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/tokens.txt';
  Config.ModelConfig.Debug := False;

  WaveFilename := './sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20/test_wavs/0.wav';

  Recognizer := TSherpaOnnxOnlineRecognizer.Create(Config);

  Stream := Recognizer.CreateStream();

  Wave := SherpaOnnxReadWave(WaveFilename);

  Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);
  Stream.InputFinished();

  while (Recognizer.IsReady(Stream)) do
    Recognizer.Decode(Stream);

  RecognitionResult := Recognizer.GetResult(Stream);
  WriteLn(RecognitionResult.ToString);

  FreeAndNil(Stream);
  FreeAndNil(Recognizer);
end.
