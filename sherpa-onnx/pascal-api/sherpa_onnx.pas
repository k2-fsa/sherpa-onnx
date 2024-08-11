{ Copyright (c)  2024  Xiaomi Corporation }

unit sherpa_onnx;

{$mode objfpc}

interface

type
  TSherpaOnnxWave = record
    Samples: array of Single; { normalized to the range [-1, 1] }
    SampleRate: Integer;
  end;

{ It supports reading a single channel wave with 16-bit encoded samples.
  Samples are normalized to the range [-1, 1].
}
function SherpaOnnxReadWave(Filename: string): TSherpaOnnxWave;

implementation

uses
  ctypes;

const
  {See https://www.freepascal.org/docs-html/prog/progap7.html}

  {$IFDEF WINDOWS}
    SherpaOnnxLibName = 'sherpa-onnx-c-api.dll';
  {$ENDIF}

  {$IFDEF DARWIN}
    SherpaOnnxLibName = 'sherpa-onnx-c-api';
    {$linklib sherpa-onnx-c-api}
  {$ENDIF}

  {$IFDEF LINUX}
    SherpaOnnxLibName = 'libsherpa-onnx-c-api.so';
  {$ENDIF}

type
  SherpaOnnxWave = record
    Samples: pcfloat;
    SampleRate: cint32;
    NumSamples: cint32;
  end;

  PSherpaOnnxWave = ^SherpaOnnxWave;

function SherpaOnnxReadWaveWrapper(Filename: PAnsiChar): PSherpaOnnxWave; cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxReadWave';

procedure SherpaOnnxFreeWaveWrapper(P: PSherpaOnnxWave); cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxFreeWave';

function SherpaOnnxReadWave(Filename: string): TSherpaOnnxWave;
var
  AnsiFilename: AnsiString;
  PFilename: PAnsiChar;
  PWave: PSherpaOnnxWave;
  I: Integer;
begin
  AnsiFilename := Filename;
  PFilename := PAnsiChar(AnsiFilename);
  PWave := SherpaOnnxReadWaveWrapper(PFilename);

  SetLength(Result.Samples, PWave^.NumSamples);

  Result.SampleRate := PWave^.SampleRate;

  for I := Low(Result.Samples) to High(Result.Samples) do
    Result.Samples[i] := PWave^.Samples[i];

  SherpaOnnxFreeWaveWrapper(PWave);
end;

end.
