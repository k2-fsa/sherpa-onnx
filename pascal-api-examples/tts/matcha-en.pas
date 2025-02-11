{ Copyright (c)  2025  Xiaomi Corporation }
program matcha_en;
{
This file shows how to use the text to speech API of sherpa-onnx
with MatchaTTS models.

It generates speech from text and saves it to a wave file.

If you want to play it while it is generating, please see
./matcha-en-playback.pas
}

{$mode objfpc}

uses
  SysUtils,
  sherpa_onnx;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Matcha.AcousticModel := './matcha-icefall-en_US-ljspeech/model-steps-3.onnx';
  Config.Model.Matcha.Vocoder := './hifigan_v2.onnx';
  Config.Model.Matcha.Tokens := './matcha-icefall-en_US-ljspeech/tokens.txt';
  Config.Model.Matcha.DataDir := './matcha-icefall-en_US-ljspeech/espeak-ng-data';
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

  Text := 'Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.';

  Audio :=  Tts.Generate(Text, SpeakerId, Speed);
  SherpaOnnxWriteWave('./matcha-en.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./matcha-en.wav');

  FreeAndNil(Tts);
end.

