{ Copyright (c)  2024  Xiaomi Corporation }

{
This file shows how to use a streaming Paraformer model to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program paraformer;

{$mode objfpc}

uses
  sherpa_onnx,
  DateUtils,
  SysUtils;

var
  Config: TSherpaOnnxOnlineRecognizerConfig;
  Recognizer: TSherpaOnnxOnlineRecognizer;
  Stream: TSherpaOnnxOnlineStream;
  RecognitionResult: TSherpaOnnxOnlineRecognizerResult;
  Wave: TSherpaOnnxWave;
  WaveFilename: AnsiString;
  TailPaddings: array of Single;

  Start: TDateTime;
  Stop: TDateTime;

  Elapsed: Single;
  Duration: Single;
  RealTimeFactor: Single;
begin
  Initialize(Config);

  {Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
  to download model files used in this file.}
  Config.ModelConfig.Paraformer.Encoder := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx';
  Config.ModelConfig.Paraformer.Decoder := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx';
  Config.ModelConfig.Tokens := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt';

  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 1;
  Config.ModelConfig.Debug := False;

  WaveFilename := './sherpa-onnx-streaming-paraformer-bilingual-zh-en/test_wavs/2.wav';

  Wave := SherpaOnnxReadWave(WaveFilename);

  Recognizer := TSherpaOnnxOnlineRecognizer.Create(Config);

  Start := Now;

  Stream := Recognizer.CreateStream();

  Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);

  SetLength(TailPaddings, Round(Wave.SampleRate * 0.5)); {0.5 seconds of padding}
  Stream.AcceptWaveform(TailPaddings, Wave.SampleRate);

  Stream.InputFinished();

  while Recognizer.IsReady(Stream) do
    Recognizer.Decode(Stream);

  RecognitionResult := Recognizer.GetResult(Stream);

  Stop := Now;

  Elapsed := MilliSecondsBetween(Stop, Start) / 1000;
  Duration := Length(Wave.Samples) / Wave.SampleRate;
  RealTimeFactor := Elapsed / Duration;

  WriteLn(RecognitionResult.ToString);
  WriteLn(Format('NumThreads %d', [Config.ModelConfig.NumThreads]));
  WriteLn(Format('Elapsed %.3f s', [Elapsed]));
  WriteLn(Format('Wave duration %.3f s', [Duration]));
  WriteLn(Format('RTF = %.3f/%.3f = %.3f', [Elapsed, Duration, RealTimeFactor]));

  {Free resources to avoid memory leak.

  Note: You don't need to invoke them for this simple script.
  However, you have to invoke them in your own large/complex project.
  }
  FreeAndNil(Stream);
  FreeAndNil(Recognizer);
end.
