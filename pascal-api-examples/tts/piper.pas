{ Copyright (c)  2024  Xiaomi Corporation }
program piper;
{
This file shows how to use the text to speech API of sherpa-onnx
with Piper models.

It generates speech from text and saves it to a wave file.

If you want to play it while it is generating, please see
./piper-playback.pas
}

{$mode objfpc}

uses
  SysUtils,
  sherpa_onnx;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Vits.Model := './vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx';
  Config.Model.Vits.Tokens := './vits-piper-en_US-libritts_r-medium/tokens.txt';
  Config.Model.Vits.DataDir := './vits-piper-en_US-libritts_r-medium/espeak-ng-data';
  Config.Model.NumThreads := 1;
  Config.Model.Debug := False;
  Config.MaxNumSentences := 1;

  Result := TSherpaOnnxOfflineTts.Create(Config);
end;

var
  Tts: TSherpaOnnxOfflineTts;
  Audio: TSherpaOnnxGeneratedAudio;

  Text: AnsiString;
  Speed: Single = 1.0;  {Use a larger value to speak faster}
  SpeakerId: Integer = 0;

begin
  Tts := GetOfflineTts;

  WriteLn('There are ', Tts.GetNumSpeakers, ' speakers');

  Text := 'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

  Audio :=  Tts.Generate(Text, SpeakerId, Speed);
  SherpaOnnxWriteWave('./libritts_r-generated.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./libritts_r-generated.wav');

  FreeAndNil(Tts);
end.

