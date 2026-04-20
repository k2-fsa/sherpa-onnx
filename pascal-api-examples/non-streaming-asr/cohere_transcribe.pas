{ Copyright (c)  2026  Xiaomi Corporation }

{
This file shows how to use a non-streaming Cohere Transcribe model
to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program cohere_transcribe;

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

  Config.ModelConfig.CohereTranscribe.Encoder := './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/encoder.int8.onnx';
  Config.ModelConfig.CohereTranscribe.Decoder := './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/decoder.int8.onnx';
  Config.ModelConfig.CohereTranscribe.UsePunct := True;
  Config.ModelConfig.CohereTranscribe.UseItn := True;
  Config.ModelConfig.Tokens := './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/tokens.txt';
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 1;
  Config.ModelConfig.Debug := True;

  WaveFilename := './sherpa-onnx-cohere-transcribe-14-lang-int8-2026-04-01/test_wavs/en.wav';

  Wave := SherpaOnnxReadWave(WaveFilename);

  Recognizer := TSherpaOnnxOfflineRecognizer.Create(Config);
  Stream := Recognizer.CreateStream();
  Stream.SetOption('language', 'en');
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

  FreeAndNil(Stream);
  FreeAndNil(Recognizer);
end.
