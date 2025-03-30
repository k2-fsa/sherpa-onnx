{ Copyright (c)  2025  Xiaomi Corporation }
{
This file shows how to use the speech enhancement API from sherpa-onnx

Please first download files used in this script before you run it.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/gtcrn_simple.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav
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
begin
  Wave := SherpaOnnxReadWave('./inp_16k.wav');

  Initialize(Config);

  Config.Model.Gtcrn.Model := './gtcrn_simple.onnx';
  Config.Model.NumThreads:= 1;
  Config.Model.Debug:= True;
  Config.Model.Provider:= 'cpu';

  Sd := TSherpaOnnxOfflineSpeechDenoiser.Create(Config);

  Audio := Sd.Run(Wave.Samples, Wave.SampleRate);

  SherpaOnnxWriteWave('./enhanced-16k.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./enhanced-16k.wav');

  FreeAndNil(Sd);
end.

