{ Copyright (c)  2025  Xiaomi Corporation }
program kokoro_en;
{
This file shows how to use the text to speech API of sherpa-onnx
with Kokoro TTS models.

It generates speech from text and saves it to a wave file.

If you want to play it while it is generating, please see
./kokoro-en-playback.pas
}

{$mode objfpc}

uses
  SysUtils,
  sherpa_onnx;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Kokoro.Model := './kokoro-en-v0_19/model.onnx';
  Config.Model.Kokoro.Voices := './kokoro-en-v0_19/voices.bin';
  Config.Model.Kokoro.Tokens := './kokoro-en-v0_19/tokens.txt';
  Config.Model.Kokoro.DataDir := './kokoro-en-v0_19/espeak-ng-data';
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
  SpeakerId: Integer = 8;

begin
  Tts := GetOfflineTts;

  WriteLn('There are ', Tts.GetNumSpeakers, ' speakers');

  Text := 'Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.';

  Audio :=  Tts.Generate(Text, SpeakerId, Speed);
  SherpaOnnxWriteWave('./kokoro-en-8.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./kokoro-en-8.wav');

  FreeAndNil(Tts);
end.

