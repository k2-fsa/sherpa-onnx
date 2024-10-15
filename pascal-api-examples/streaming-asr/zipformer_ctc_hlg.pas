{ Copyright (c)  2024  Xiaomi Corporation }

{
This file shows how to use a streaming Zipformer CTC model
with HLG to decode files.

You can download the model files from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
}

program zipformer_ctc_hlg;

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
  Config.ModelConfig.Zipformer2Ctc.Model := './sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/ctc-epoch-30-avg-3-chunk-16-left-128.int8.onnx';
  Config.ModelConfig.Tokens := './sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/tokens.txt';
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 1;
  Config.ModelConfig.Debug := True;
  Config.CtcFstDecoderConfig.Graph := './sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/HLG.fst';

  WaveFilename := './sherpa-onnx-streaming-zipformer-ctc-small-2024-03-18/test_wavs/8k.wav';

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
