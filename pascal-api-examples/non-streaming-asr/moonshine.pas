{ Copyright (c)  2024  Xiaomi Corporation }

{
This file shows how to use a non-streaming Moonshine model
to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program moonshine;

{$mode objfpc}

uses
  sherpa_onnx,
  DateUtils,
  SysUtils;

var
  Wave: TSherpaOnnxWave;
  WaveFilename: AnsiString;

  Config: TSherpaOnnxOfflineRecognizerConfig;
  Recognizer: TSherpaOnnxOfflineRecognizer;
  Stream: TSherpaOnnxOfflineStream;
  RecognitionResult: TSherpaOnnxOfflineRecognizerResult;

  Start: TDateTime;
  Stop: TDateTime;

  Elapsed: Single;
  Duration: Single;
  RealTimeFactor: Single;
begin
  Initialize(Config);

  Config.ModelConfig.Moonshine.Preprocessor := './sherpa-onnx-moonshine-tiny-en-int8/preprocess.onnx';
  Config.ModelConfig.Moonshine.Encoder := './sherpa-onnx-moonshine-tiny-en-int8/encode.int8.onnx';
  Config.ModelConfig.Moonshine.UncachedDecoder := './sherpa-onnx-moonshine-tiny-en-int8/uncached_decode.int8.onnx';
  Config.ModelConfig.Moonshine.CachedDecoder := './sherpa-onnx-moonshine-tiny-en-int8/cached_decode.int8.onnx';

  Config.ModelConfig.Tokens := './sherpa-onnx-moonshine-tiny-en-int8/tokens.txt';
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 1;
  Config.ModelConfig.Debug := False;

  WaveFilename := './sherpa-onnx-moonshine-tiny-en-int8/test_wavs/0.wav';

  Wave := SherpaOnnxReadWave(WaveFilename);

  Recognizer := TSherpaOnnxOfflineRecognizer.Create(Config);
  Stream := Recognizer.CreateStream();
  Start := Now;

  Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);
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
