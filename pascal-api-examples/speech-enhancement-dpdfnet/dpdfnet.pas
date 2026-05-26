{ Copyright (c)  2026  Xiaomi Corporation }
{
This file shows how to use the offline speech enhancement API from sherpa-onnx
with a DPDFNet model.

Please first download files used in this script before you run it.

wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/dpdfnet_baseline.onnx
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/speech-enhancement-models/inp_16k.wav

Use dpdfnet_baseline.onnx, dpdfnet2.onnx, dpdfnet4.onnx, or dpdfnet8.onnx
for 16 kHz downstream ASR or speech recognition.
Use dpdfnet2_48khz_hr.onnx for 48 kHz enhancement output.
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
  Config.Model.DpdfNet.Model := './dpdfnet_baseline.onnx';
  Config.Model.NumThreads:= 1;
  Config.Model.Debug:= True;
  Config.Model.Provider:= 'cpu';

  Sd := TSherpaOnnxOfflineSpeechDenoiser.Create(Config);

  Audio := Sd.Run(Wave.Samples, Wave.SampleRate);

  SherpaOnnxWriteWave('./enhanced-dpdfnet.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./enhanced-dpdfnet.wav');

  FreeAndNil(Sd);
end.
