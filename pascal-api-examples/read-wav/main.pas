program main;

{$mode objfpc}

uses
  sherpa_onnx;

var
  Wave: TSherpaOnnxWave;
  S: Single;
  I: Integer;
begin
  Wave := SherpaOnnxReadWaveWrapper('./lei-jun-test.wav');
  WriteLn('info ', Wave.SampleRate, ' ', Length(Wave.Samples));
  S := 0;
  for i := Low(Wave.Samples) to High(Wave.Samples) do
    S += Wave.Samples[i];

  WriteLn('sum is ', S);
end.
