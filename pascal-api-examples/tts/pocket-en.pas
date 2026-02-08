{ Copyright (c)  2026  Xiaomi Corporation }
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
  Config.Model.Pocket.LmFlow := './sherpa-onnx-pocket-tts-int8-2026-01-26/lm_flow.int8.onnx';
  Config.Model.Pocket.LmMain := './sherpa-onnx-pocket-tts-int8-2026-01-26/lm_main.int8.onnx';
  Config.Model.Pocket.Encoder := './sherpa-onnx-pocket-tts-int8-2026-01-26/encoder.onnx';
  Config.Model.Pocket.Decoder := './sherpa-onnx-pocket-tts-int8-2026-01-26/decoder.int8.onnx';
  Config.Model.Pocket.TextConditioner := './sherpa-onnx-pocket-tts-int8-2026-01-26/text_conditioner.onnx';
  Config.Model.Pocket.VocabJson := './sherpa-onnx-pocket-tts-int8-2026-01-26/vocab.json';
  Config.Model.Pocket.TokenScoresJson := './sherpa-onnx-pocket-tts-int8-2026-01-26/token_scores.json';
  Config.Model.NumThreads := 2;
  Config.Model.Debug := True;
  Config.MaxNumSentences := 1;

  Result := TSherpaOnnxOfflineTts.Create(Config);
end;

var
  Tts: TSherpaOnnxOfflineTts;
  GenerationConfig: TSherpaOnnxGenerationConfig;
  Wave: TSherpaOnnxWave;
  WaveFilename: AnsiString;
  Audio: TSherpaOnnxGeneratedAudio;

  Text: AnsiString;
  Speed: Single = 1.0;  {Use a larger value to speak faster}
  SpeakerId: Integer = 0;

begin
  Tts := GetOfflineTts;

  WriteLn('There are ', Tts.GetNumSpeakers, ' speakers');

  Text := 'Friends fell out often because life was changing so fast. The easiest thing in the world was to lose touch with someone.';

  WaveFilename := './sherpa-onnx-pocket-tts-int8-2026-01-26/test_wavs/bria.wav';
  Wave := SherpaOnnxReadWave(WaveFilename);
  GenerationConfig.ReferenceAudio := Wave.Samples;
  GenerationConfig.ReferenceAudioLen := Length(Wave.Samples);
  GenerationConfig.ReferenceSampleRate := Wave.SampleRate;

  Audio :=  Tts.Generate(Text, GenerationConfig, NIL, NIL);
  SherpaOnnxWriteWave('./pocket-tts-en.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./pocket-tts-en.wav');

  FreeAndNil(Tts);
end.

