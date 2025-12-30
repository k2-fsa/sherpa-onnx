{ Copyright (c)  2025  Xiaomi Corporation }
program kitten_en;
{
This file shows how to use the text to speech API of sherpa-onnx
with Kitten TTS models.

It generates speech from text and saves it to a wave file.

If you want to play it while it is generating, please see
./kitten-en-playback.pas
}

{$mode objfpc}

uses
  SysUtils,
  sherpa_onnx;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Kitten.Model := './kitten-nano-en-v0_1-fp16/model.fp16.onnx';
  Config.Model.Kitten.Voices := './kitten-nano-en-v0_1-fp16/voices.bin';
  Config.Model.Kitten.Tokens := './kitten-nano-en-v0_1-fp16/tokens.txt';
  Config.Model.Kitten.DataDir := './kitten-nano-en-v0_1-fp16/espeak-ng-data';
  Config.Model.NumThreads := 2;
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

  Text := 'Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.';

  Audio :=  Tts.Generate(Text, SpeakerId, Speed);
  SherpaOnnxWriteWave('./kitten-en-0.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./kitten-en-0.wav');

  FreeAndNil(Tts);
end.

