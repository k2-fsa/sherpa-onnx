{ Copyright (c)  2026  Xiaomi Corporation }
{
This file shows how to use the streaming speech enhancement API from
sherpa-onnx with VoxLock models and a voice sample.

VoxLock is a target-speaker voice isolation model. It keeps the voice of one
speaker, e.g., the caller in a phone call, and removes the other voices.

The model is from https://github.com/arbazsiddiqui/voxlock

This example provides a voice sample of the speaker to keep, so VoxLock
isolates that speaker from the first sample. The voice sample must have at
least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
loudest 3 seconds within its first 30 seconds. See
../streaming-speech-enhancement-voxlock if you don't have a voice sample.

Please first download files used in this script before you run it.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/sherpa-onnx-voxlock-int8.tar.bz2
tar xvf sherpa-onnx-voxlock-int8.tar.bz2
}
program main;

{$mode delphi}

uses
  sherpa_onnx,
  SysUtils;

var
  Wave: TSherpaOnnxWave;

  Config: TSherpaOnnxOnlineSpeechDenoiserConfig;
  Sd: TSherpaOnnxOnlineSpeechDenoiser;
  Audio: TSherpaOnnxDenoisedAudio;
  Chunk: array of Single;
  Enhanced: array of Single;
  StartIndex: Integer;
  N: Integer;
  NewLength: Integer;
  StartMs: QWord;
  ElapsedSeconds, Duration: Double;
begin
  Wave := SherpaOnnxReadWave('./sherpa-onnx-voxlock-int8/test_wavs/en-combined.wav');

  Initialize(Config);
  Config.Model.VoxLock.EnrollModel := './sherpa-onnx-voxlock-int8/enroll.onnx';
  Config.Model.VoxLock.ChunkModel := './sherpa-onnx-voxlock-int8/chunk_int8.onnx';
  Config.Model.VoxLock.VoiceSample := './sherpa-onnx-voxlock-int8/test_wavs/en-voice-sample-f.wav';
  Config.Model.NumThreads:= 1;
  Config.Model.Debug:= True;
  Config.Model.Provider:= 'cpu';

  Sd := TSherpaOnnxOnlineSpeechDenoiser.Create(Config);

  SetLength(Enhanced, 0);
  StartIndex := 0;
  StartMs := GetTickCount64;
  while StartIndex < Length(Wave.Samples) do
    begin
      N := Sd.GetFrameShiftInSamples;
      if StartIndex + N > Length(Wave.Samples) then
        N := Length(Wave.Samples) - StartIndex;

      Chunk := Copy(Wave.Samples, StartIndex, N);
      Audio := Sd.Run(Chunk, Wave.SampleRate);
      NewLength := Length(Enhanced) + Length(Audio.Samples);
      SetLength(Enhanced, NewLength);
      if Length(Audio.Samples) > 0 then
        Move(Audio.Samples[0], Enhanced[NewLength - Length(Audio.Samples)],
          Length(Audio.Samples) * SizeOf(Single));
      Inc(StartIndex, N);
    end;

  Audio := Sd.Flush;
  NewLength := Length(Enhanced) + Length(Audio.Samples);
  SetLength(Enhanced, NewLength);
  if Length(Audio.Samples) > 0 then
    Move(Audio.Samples[0], Enhanced[NewLength - Length(Audio.Samples)],
      Length(Audio.Samples) * SizeOf(Single));
  ElapsedSeconds := (GetTickCount64 - StartMs) / 1000.0;

  SherpaOnnxWriteWave('./voxlock-enhanced-online-with-voice-sample-16k.wav', Enhanced,
    Sd.GetSampleRate);

  Duration := Length(Wave.Samples) / Wave.SampleRate;
  WriteLn('Saved to ./voxlock-enhanced-online-with-voice-sample-16k.wav');
  WriteLn(Format('Elapsed seconds: %.3f', [ElapsedSeconds]));
  WriteLn(Format('Audio duration in seconds: %.3f', [Duration]));
  WriteLn(Format('RTF: %.3f/%.3f = %.3f',
    [ElapsedSeconds, Duration, ElapsedSeconds / Duration]));

  FreeAndNil(Sd);
end.
