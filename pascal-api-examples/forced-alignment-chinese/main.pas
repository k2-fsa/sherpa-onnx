{ Copyright (c)  2026  Xiaomi Corporation }

{
Forced alignment of Chinese speech with the sherpa-onnx Pascal API.

It aligns the test wav of the zh-aishell3-pinyin model package with its
transcript and writes a Praat TextGrid. The model directory, its
pronunciation lexicon and the test wav are hard-coded below.

How to download the model (the package contains the model files, the
lexicon and the test wav used here):

  wget https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
  tar xf zh-aishell3-pinyin.tar.bz2

How to run:

  ./run.sh

Output:

  zh.TextGrid in the current directory. Open it with Praat
  (https://www.fon.hum.uva.nl/praat/) to visualize the phone and word
  boundaries.

See also ../forced-alignment-english and the Python examples
python-api-examples/forced-alignment-chinese.py.
}

program main;

{$mode objfpc}

uses
  sherpa_onnx,
  DateUtils,
  SysUtils;

var
  ModelDir: AnsiString;
  WaveFilename: AnsiString;
  Text: AnsiString;

  Config: TSherpaOnnxForcedAlignerConfig;
  Aligner: TSherpaOnnxForcedAligner;
  Stream: TSherpaOnnxOfflineStream;
  Wave: TSherpaOnnxWave;
  Res: TSherpaOnnxForcedAlignmentResult;

  StartInit: TDateTime;
  StopInit: TDateTime;
  Start: TDateTime;
  Stop: TDateTime;

  InitSeconds: Single;
  ElapsedSeconds: Single;
  AudioDuration: Single;
  RealTimeFactor: Single;

  I: Integer;
  F: TextFile;
begin
  ModelDir := './zh-aishell3-pinyin';
  WaveFilename := ModelDir + '/zh.wav';
  Text := '不要问你的国家能为你做什么，而要问你能为你的国家做什么。';

  Initialize(Config);
  Config.ModelDir := ModelDir;
  Config.Lexicon := ModelDir + '/lexicon.txt';

  Wave := SherpaOnnxReadWave(WaveFilename);

  StartInit := Now;
  Aligner := TSherpaOnnxForcedAligner.Create(Config);
  Stream := Aligner.CreateStream();
  StopInit := Now;

  Stream.AcceptWaveform(Wave.Samples, Wave.SampleRate);

  Start := Now;
  Res := Aligner.Compute(Stream, Text);
  Stop := Now;

  { tiers[0] is always the "phones" tier, tiers[1] the "words" tier }
  WriteLn('--- ', Res.Tiers[0].Name, ' (',
    Length(Res.Tiers[0].Intervals), ' intervals) ---');
  for I := Low(Res.Tiers[0].Intervals) to High(Res.Tiers[0].Intervals) do
    WriteLn(Format('%.2f - %.2f  %s',
      [Res.Tiers[0].Intervals[I].StartTime,
       Res.Tiers[0].Intervals[I].EndTime,
       Res.Tiers[0].Intervals[I].Text]));

  WriteLn('--- ', Res.Tiers[1].Name, ' (',
    Length(Res.Tiers[1].Intervals), ' intervals) ---');
  for I := Low(Res.Tiers[1].Intervals) to High(Res.Tiers[1].Intervals) do
    WriteLn(Format('%.2f - %.2f  %s',
      [Res.Tiers[1].Intervals[I].StartTime,
       Res.Tiers[1].Intervals[I].EndTime,
       Res.Tiers[1].Intervals[I].Text]));

  AssignFile(F, 'zh.TextGrid');
  Rewrite(F);
  Write(F, Res.ToTextGrid);
  CloseFile(F);
  WriteLn('wrote zh.TextGrid');

  { time statistics, like sherpa-onnx-fa prints }
  InitSeconds := MilliSecondsBetween(StopInit, StartInit) / 1000;
  ElapsedSeconds := MilliSecondsBetween(Stop, Start) / 1000;
  AudioDuration := Length(Wave.Samples) / Wave.SampleRate;
  RealTimeFactor := ElapsedSeconds / AudioDuration;

  WriteLn(Format('init is done in %.3f s', [InitSeconds]));
  WriteLn(Format('Elapsed seconds: %.3f s', [ElapsedSeconds]));
  WriteLn(Format('Real time factor (RTF): %.3f / %.3f = %.3f',
    [ElapsedSeconds, AudioDuration, RealTimeFactor]));
  WriteLn(Format('phones: %d, words: %d',
    [Length(Res.Tiers[0].Intervals), Length(Res.Tiers[1].Intervals)]));

  {Free resources to avoid memory leak.

  Note: You don't need to invoke them for this simple script.
  However, you have to invoke them in your own large/complex project.
  }
  FreeAndNil(Stream);
  FreeAndNil(Aligner);
end.
