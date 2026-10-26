{ Copyright (c)  2026  Xiaomi Corporation }
{
This file shows how to use the speech enhancement API from sherpa-onnx with
VoxLock models and a voice sample.

VoxLock is a target-speaker voice isolation model. It keeps the voice of one
speaker, e.g., the caller in a phone call, and removes the other voices.

The model is from https://github.com/arbazsiddiqui/voxlock

This example provides a voice sample of the speaker to keep, so VoxLock
isolates that speaker from the first sample. The voice sample must have at
least 3 seconds of the speaker's voice; from a longer file, VoxLock uses the
loudest 3 seconds within its first 30 seconds. See
../speech-enhancement-voxlock if you don't have a voice sample.

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

  Config: TSherpaOnnxOfflineSpeechDenoiserConfig;
  Sd: TSherpaOnnxOfflineSpeechDenoiser;
  Audio: TSherpaOnnxDenoisedAudio;
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

  Sd := TSherpaOnnxOfflineSpeechDenoiser.Create(Config);

  StartMs := GetTickCount64;
  Audio := Sd.Run(Wave.Samples, Wave.SampleRate);
  ElapsedSeconds := (GetTickCount64 - StartMs) / 1000.0;

  SherpaOnnxWriteWave('./voxlock-enhanced-with-voice-sample-16k.wav', Audio.Samples,
    Audio.SampleRate);

  Duration := Length(Wave.Samples) / Wave.SampleRate;
  WriteLn('Saved to ./voxlock-enhanced-with-voice-sample-16k.wav');
  WriteLn(Format('Elapsed seconds: %.3f', [ElapsedSeconds]));
  WriteLn(Format('Audio duration in seconds: %.3f', [Duration]));
  WriteLn(Format('RTF: %.3f/%.3f = %.3f',
    [ElapsedSeconds, Duration, ElapsedSeconds / Duration]));

  FreeAndNil(Sd);
end.
