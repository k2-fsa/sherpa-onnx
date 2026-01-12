{ Copyright (c)  2026  Xiaomi Corporation }

{
This file shows how to use a non-streaming FunASR Nano model
to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program funasr_nano;

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

  Config.ModelConfig.FunAsrNano.EncoderAdaptor := './sherpa-onnx-funasr-nano-int8-2025-12-30/encoder_adaptor.int8.onnx';
  Config.ModelConfig.FunAsrNano.LLM := './sherpa-onnx-funasr-nano-int8-2025-12-30/llm.int8.onnx';
  Config.ModelConfig.FunAsrNano.Embedding := './sherpa-onnx-funasr-nano-int8-2025-12-30/embedding.int8.onnx';
  Config.ModelConfig.FunAsrNano.Tokenizer := './sherpa-onnx-funasr-nano-int8-2025-12-30/Qwen3-0.6B';
  Config.ModelConfig.Tokens := '';
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 2;
  Config.ModelConfig.Debug := True;

  WaveFilename := './sherpa-onnx-funasr-nano-int8-2025-12-30/test_wavs/lyrics.wav';

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
