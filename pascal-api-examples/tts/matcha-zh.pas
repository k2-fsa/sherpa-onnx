{ Copyright (c)  2025  Xiaomi Corporation }
program matcha_zh;
{
This file shows how to use the text to speech API of sherpa-onnx
with MatchaTTS models.

It generates speech from text and saves it to a wave file.

If you want to play it while it is generating, please see
./matcha-zh-playback.pas
}

{$mode objfpc}

uses
  SysUtils,
  sherpa_onnx;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Matcha.AcousticModel := './matcha-icefall-zh-baker/model-steps-3.onnx';
  Config.Model.Matcha.Vocoder := './hifigan_v2.onnx';
  Config.Model.Matcha.Lexicon := './matcha-icefall-zh-baker/lexicon.txt';
  Config.Model.Matcha.Tokens := './matcha-icefall-zh-baker/tokens.txt';
  Config.Model.Matcha.DictDir := './matcha-icefall-zh-baker/dict';
  Config.Model.NumThreads := 1;
  Config.Model.Debug := False;
  Config.RuleFsts := './matcha-icefall-zh-baker/phone.fst,./matcha-icefall-zh-baker/date.fst,./matcha-icefall-zh-baker/number.fst';
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

  Text := '某某银行的副行长和一些行政领导表示，他们去过长江和长白山; 经济不断增长。2024年12月31号，拨打110或者18920240511。123456块钱。';

  Audio :=  Tts.Generate(Text, SpeakerId, Speed);
  SherpaOnnxWriteWave('./matcha-zh.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./matcha-zh.wav');

  FreeAndNil(Tts);
end.

