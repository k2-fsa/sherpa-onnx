{ Copyright (c)  2026  Xiaomi Corporation }

{
This file shows how to use a non-streaming Qwen3 ASR model
to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program qwen3_asr;

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

  Config.ModelConfig.Qwen3Asr.ConvFrontend := './sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/conv_frontend.onnx';
  Config.ModelConfig.Qwen3Asr.Encoder := './sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/encoder.int8.onnx';
  Config.ModelConfig.Qwen3Asr.Decoder := './sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/decoder.int8.onnx';
  Config.ModelConfig.Qwen3Asr.Tokenizer := './sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/tokenizer';
  Config.ModelConfig.Tokens := '';
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 3;
  Config.ModelConfig.Debug := True;

  WaveFilename := './sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/test_wavs/raokouling.wav';

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
