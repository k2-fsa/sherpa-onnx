{ Copyright (c)  2024  Xiaomi Corporation }

{
This file shows how to use a non-streaming Moonshine model
with silero VAD to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program vad_with_moonshine;

{$mode objfpc}

uses
  sherpa_onnx,
  SysUtils;

function CreateVad(): TSherpaOnnxVoiceActivityDetector;
var
  Config: TSherpaOnnxVadModelConfig;

  SampleRate: Integer;
  WindowSize: Integer;
begin
  Initialize(Config);

  SampleRate := 16000; {Please don't change it unless you know the details}
  WindowSize := 512; {Please don't change it unless you know the details}

  Config.SileroVad.Model := './silero_vad.onnx';
  Config.SileroVad.MinSpeechDuration := 0.5;
  Config.SileroVad.MinSilenceDuration := 0.5;
  Config.SileroVad.Threshold := 0.5;
  Config.SileroVad.WindowSize := WindowSize;
  Config.NumThreads:= 1;
  Config.Debug:= True;
  Config.Provider:= 'cpu';
  Config.SampleRate := SampleRate;

  Result := TSherpaOnnxVoiceActivityDetector.Create(Config, 30);
end;

function CreateOfflineRecognizer(): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
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

  Result := TSherpaOnnxOfflineRecognizer.Create(Config);
end;

var
  Wave: TSherpaOnnxWave;

  Recognizer: TSherpaOnnxOfflineRecognizer;
  Vad: TSherpaOnnxVoiceActivityDetector;

  Offset: Integer;
  WindowSize: Integer;
  SpeechSegment: TSherpaOnnxSpeechSegment;

  Start: Single;
  Duration: Single;

  Stream: TSherpaOnnxOfflineStream;
  RecognitionResult: TSherpaOnnxOfflineRecognizerResult;
begin
  Vad := CreateVad();
  Recognizer := CreateOfflineRecognizer();

  Wave := SherpaOnnxReadWave('./Obama.wav');
  if Wave.SampleRate <> Vad.Config.SampleRate then
    begin
      WriteLn(Format('Expected sample rate: %d. Given: %d',
        [Vad.Config.SampleRate, Wave.SampleRate]));

      Exit;
    end;

  WindowSize := Vad.Config.SileroVad.WindowSize;
  Offset := 0;
  while Offset + WindowSize <= Length(Wave.Samples) do
    begin
      Vad.AcceptWaveform(Wave.Samples, Offset, WindowSize);
      Offset += WindowSize;

      while not Vad.IsEmpty do
        begin
          SpeechSegment := Vad.Front();
          Vad.Pop();
          Stream := Recognizer.CreateStream();

          Stream.AcceptWaveform(SpeechSegment.Samples, Wave.SampleRate);
          Recognizer.Decode(Stream);
          RecognitionResult := Recognizer.GetResult(Stream);

          Start := SpeechSegment.Start / Wave.SampleRate;
          Duration := Length(SpeechSegment.Samples) / Wave.SampleRate;
          WriteLn(Format('%.3f -- %.3f %s',
            [Start, Start + Duration, RecognitionResult.Text]));

          FreeAndNil(Stream);
        end;
    end;

  Vad.Flush;

  while not Vad.IsEmpty do
    begin
      SpeechSegment := Vad.Front();
      Vad.Pop();
      Stream := Recognizer.CreateStream();

      Stream.AcceptWaveform(SpeechSegment.Samples, Wave.SampleRate);
      Recognizer.Decode(Stream);
      RecognitionResult := Recognizer.GetResult(Stream);

      Start := SpeechSegment.Start / Wave.SampleRate;
      Duration := Length(SpeechSegment.Samples) / Wave.SampleRate;
      WriteLn(Format('%.3f -- %.3f %s',
        [Start, Start + Duration, RecognitionResult.Text]));

      FreeAndNil(Stream);
    end;

  FreeAndNil(Recognizer);
  FreeAndNil(Vad);
end.
