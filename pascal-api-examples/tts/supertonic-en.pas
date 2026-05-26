{ Copyright (c)  2026  Xiaomi Corporation }
program supertonic_en;
{
This file shows how to use the text to speech API of sherpa-onnx
with Supertonic TTS models.

It generates speech from text and saves it to a wave file.

Please visit
https://k2-fsa.github.io/sherpa/onnx/tts/supertonic.html
to download the model.
}

{$mode objfpc}

uses
  SysUtils,
  sherpa_onnx;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Supertonic.DurationPredictor := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/duration_predictor.int8.onnx';
  Config.Model.Supertonic.TextEncoder := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/text_encoder.int8.onnx';
  Config.Model.Supertonic.VectorEstimator := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vector_estimator.int8.onnx';
  Config.Model.Supertonic.Vocoder := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/vocoder.int8.onnx';
  Config.Model.Supertonic.TtsJson := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/tts.json';
  Config.Model.Supertonic.UnicodeIndexer := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/unicode_indexer.bin';
  Config.Model.Supertonic.VoiceStyle := './sherpa-onnx-supertonic-3-tts-int8-2026-05-11/voice.bin';
  Config.Model.NumThreads := 2;
  Config.Model.Debug := True;
  Config.MaxNumSentences := 1;

  Result := TSherpaOnnxOfflineTts.Create(Config);
end;

var
  Tts: TSherpaOnnxOfflineTts;
  GenerationConfig: TSherpaOnnxGenerationConfig;
  Audio: TSherpaOnnxGeneratedAudio;
  Text: AnsiString;

begin
  Tts := GetOfflineTts;

  WriteLn('There are ', Tts.GetNumSpeakers, ' speakers');

  Text := 'Today as always, men fall into two groups: slaves and free men. Whoever ' +
    'does not have two-thirds of his day for himself, is a slave, whatever ' +
    'he may be: a statesman, a businessman, an official, or a scholar.';

  GenerationConfig.Sid := 6;
  GenerationConfig.NumSteps := 8;
  GenerationConfig.Speed := 1.25;
  GenerationConfig.Extra := '{"lang": "en"}';

  Audio := Tts.Generate(Text, GenerationConfig, NIL, NIL);
  SherpaOnnxWriteWave('./supertonic-tts-en.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./supertonic-tts-en.wav');

  FreeAndNil(Tts);
end.
