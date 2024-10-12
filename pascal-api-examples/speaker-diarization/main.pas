{ Copyright (c)  2024  Xiaomi Corporation }
{
This file shows how to use the Pascal API from sherpa-onnx
for speaker diarization.

Usage:

Step 1: Download a speaker segmentation model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  tar xvf sherpa-onnx-pyannote-segmentation-3-0.tar.bz2
  rm sherpa-onnx-pyannote-segmentation-3-0.tar.bz2

Step 2: Download a speaker embedding extractor model

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
for a list of available models. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx

Step 3. Download test wave files

Please visit https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-segmentation-models
for a list of available test wave files. The following is an example

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav

Step 4. Run it
}

program main;

{$mode delphi}

uses
  sherpa_onnx,
  ctypes,
  SysUtils;

function ProgressCallback(
      NumProcessedChunks: cint32;
      NumTotalChunks: cint32): cint32; cdecl;
var
  Progress: Single;
begin
  Progress := 100.0 * NumProcessedChunks / NumTotalChunks;
  WriteLn(Format('Progress: %.3f%%', [Progress]));

  Result := 0;
end;

var
  Wave: TSherpaOnnxWave;
  Config: TSherpaOnnxOfflineSpeakerDiarizationConfig;
  Sd: TSherpaOnnxOfflineSpeakerDiarization;
  Segments: TSherpaOnnxOfflineSpeakerDiarizationSegmentArray;
  I: Integer;
begin
  Wave := SherpaOnnxReadWave('./0-four-speakers-zh.wav');

  Config.Segmentation.Pyannote.Model := './sherpa-onnx-pyannote-segmentation-3-0/model.onnx';
  Config.Embedding.Model := './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx';

  {
    Since we know that there are 4 speakers in ./0-four-speakers-zh.wav, we
    set NumClusters to 4 here.
    If you don't have such information, please set NumClusters to -1.
    In that case, you have to set Config.Clustering.Threshold.
    A larger threshold leads to fewer clusters, i.e., fewer speakers.
  }
  Config.Clustering.NumClusters := 4;
  Config.Segmentation.Debug := True;
  Config.Embedding.Debug := True;

  Sd := TSherpaOnnxOfflineSpeakerDiarization.Create(Config);
  if Sd.GetHandle = nil then
    begin
      WriteLn('Please check you config');
      Exit;
    end;

  if Sd.GetSampleRate <> Wave.SampleRate then
    begin
      WriteLn(Format('Expected sample rate: %d, given: %d', [Sd.GetSampleRate, Wave.SampleRate]));
      Exit;
    end;

  {
    // If you don't want to use a callback
    Segments := Sd.Process(Wave.Samples);
  }
  Segments := Sd.Process(Wave.Samples, @ProgressCallback);

  for I := Low(Segments) to High(Segments) do
    begin
      WriteLn(Format('%.3f -- %.3f speaker_%d',
        [Segments[I].Start, Segments[I].Stop, Segments[I].Speaker]));
    end;

  FreeAndNil(Sd);
end.
