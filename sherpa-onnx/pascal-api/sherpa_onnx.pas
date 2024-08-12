{ Copyright (c)  2024  Xiaomi Corporation }

unit sherpa_onnx;

{$mode objfpc}

{$modeSwitch advancedRecords} { to support records with methods }
(* {$LongStrings ON} *)

interface

type
  TSherpaOnnxWave = record
    Samples: array of Single; { normalized to the range [-1, 1] }
    SampleRate: Integer;
  end;

  TSherpaOnnxOnlineTransducerModelConfig = record
    Encoder: AnsiString;
    Decoder: AnsiString;
    Joiner: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineParaformerModelConfig = record
    Encoder: AnsiString;
    Decoder: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineZipformer2CtcModelConfig = record
    Model: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineModelConfig = record
    Transducer: TSherpaOnnxOnlineTransducerModelConfig;
    Paraformer: TSherpaOnnxOnlineParaformerModelConfig;
    Zipformer2Ctc: TSherpaOnnxOnlineZipformer2CtcModelConfig;
    Tokens: AnsiString;
    NumThreads: Integer;
    Provider: AnsiString;
    Debug: Boolean;
    ModelType: AnsiString;
    ModelingUnit: AnsiString;
    BpeVocab: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxFeatureConfig = record
    SampleRate: Integer;
    FeatureDim: Integer;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineCtcFstDecoderConfig = record
    Graph: AnsiString;
    MaxActive: Integer;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineRecognizerConfig = record
    FeatConfig: TSherpaOnnxFeatureConfig;
    ModelConfig: TSherpaOnnxOnlineModelConfig;
    DecodingMethod: AnsiString;
    MaxActivePaths: Integer;
    EnableEndpoint: Boolean;
    Rule1MinTrailingSilence: Single;
    Rule2MinTrailingSilence: Single;
    Rule3MinUtteranceLength: Single;
    HotwordsFile: AnsiString;
    HotwordsScore: Single;
    CtcFstDecoderConfig: TSherpaOnnxOnlineCtcFstDecoderConfig;
    RuleFsts: AnsiString;
    RuleFars: AnsiString;
    BlankPenalty: Single;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineRecognizerResult = record
    Text: AnsiString;
  end;

  TSherpaOnnxOnlineRecognizer = class
  private
   handle: Pointer;
  public
    constructor Create(Config: TSherpaOnnxOnlineRecognizerConfig);
    destructor Destroy; override;
  end;

{ It supports reading a single channel wave with 16-bit encoded samples.
  Samples are normalized to the range [-1, 1].
}
function SherpaOnnxReadWave(Filename: AnsiString): TSherpaOnnxWave;

implementation

uses
  ctypes,
  SysUtils;

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

  SherpaOnnxOnlineTransducerModelConfig = record
    Encoder: PAnsiChar;
    Decoder: PAnsiChar;
    Joiner: PAnsiChar;
  end;
  SherpaOnnxOnlineParaformerModelConfig = record
    Encoder: PAnsiChar;
    Decoder: PAnsiChar;
  end;
  SherpaOnnxOnlineZipformer2CtcModelConfig = record
    Model: PAnsiChar;
  end;

  SherpaOnnxOnlineModelConfig= record
    Transducer: SherpaOnnxOnlineTransducerModelConfig;
    Paraformer: SherpaOnnxOnlineParaformerModelConfig;
    Zipformer2Ctc: SherpaOnnxOnlineZipformer2CtcModelConfig;
    Tokens: PAnsiChar;
    NumThreads: cint32;
    Provider: PAnsiChar;
    Debug: cint32;
    ModelType: PAnsiChar;
    ModelingUnit: PAnsiChar;
    BpeVocab: PAnsiChar;
  end;
  SherpaOnnxFeatureConfig = record
    SampleRate: cint32;
    FeatureDim: cint32;
  end;
  SherpaOnnxOnlineCtcFstDecoderConfig = record
    Graph: PAnsiChar;
    MaxActive: cint32;
  end;
  SherpaOnnxOnlineRecognizerConfig = record
    FeatConfig: SherpaOnnxFeatureConfig;
    ModelConfig: SherpaOnnxOnlineModelConfig;
    DecodingMethod: PAnsiChar;
    MaxActivePaths: cint32;
    EnableEndpoint: cint32;
    Rule1MinTrailingSilence: Single;
    Rule2MinTrailingSilence: Single;
    Rule3MinUtteranceLength: Single;
    HotwordsFile: PAnsiChar;
    HotwordsScore: Single;
    CtcFstDecoderConfig: SherpaOnnxOnlineCtcFstDecoderConfig;
    RuleFsts: PAnsiChar;
    RuleFars: PAnsiChar;
    BlankPenalty: Single;
  end;

  PSherpaOnnxOnlineRecognizerConfig = ^SherpaOnnxOnlineRecognizerConfig;

function SherpaOnnxCreateOnlineRecognizerWrapper(Config: PSherpaOnnxOnlineRecognizerConfig): Pointer; cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxCreateOnlineRecognizer';

procedure SherpaOnnxDestroyOnlineRecognizer(p: Pointer); cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxDestroyOnlineRecognizer';

function SherpaOnnxReadWaveWrapper(Filename: PAnsiChar): PSherpaOnnxWave; cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxReadWave';

procedure SherpaOnnxFreeWaveWrapper(P: PSherpaOnnxWave); cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxFreeWave';

function SherpaOnnxReadWave(Filename: AnsiString): TSherpaOnnxWave;
var
  PFilename: PAnsiChar;
  PWave: PSherpaOnnxWave;
  I: Integer;
begin
  PFilename := PAnsiChar(Filename);
  PWave := SherpaOnnxReadWaveWrapper(PFilename);

  SetLength(Result.Samples, PWave^.NumSamples);

  Result.SampleRate := PWave^.SampleRate;

  for I := Low(Result.Samples) to High(Result.Samples) do
    Result.Samples[I] := PWave^.Samples[I];

  SherpaOnnxFreeWaveWrapper(PWave);
end;

function TSherpaOnnxOnlineTransducerModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOnlineTransducerModelConfig(Encoder := %s, Decoder := %s, Joiner := %s)',
  [Self.Encoder, Self.Decoder, Self.Joiner]);
end;

function TSherpaOnnxOnlineParaformerModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOnlineParaformerModelConfig(Encoder := %s, Decoder := %s)',
  [Self.Encoder, Self.Decoder]);
end;

function TSherpaOnnxOnlineZipformer2CtcModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOnlineZipformer2CtcModelConfig(Model := %s)',
  [Self.Model]);
end;

function TSherpaOnnxOnlineModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOnlineModelConfig(Transducer := %s, ' +
    'Paraformer := %s,' +
    'Zipformer2Ctc := %s, ' +
    'Tokens := %s, ' +
    'NumThreads := %d, ' +
    'Provider := %s, ' +
    'Debug := %s, ' +
    'ModelType := %s, ' +
    'ModelingUnit := %s, ' +
    'BpeVocab := %s)'
    ,
  [Self.Transducer.ToString, Self.Paraformer.ToString,
   Self.Zipformer2Ctc.ToString, Self.Tokens,
   Self.NumThreads, Self.Provider, Self.Debug.ToString,
   Self.ModelType, Self.ModelingUnit, Self.BpeVocab
  ]);
end;

function TSherpaOnnxFeatureConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxFeatureConfig(SampleRate := %d, FeatureDim := %d)',
    [Self.SampleRate, Self.FeatureDim]);
end;

function TSherpaOnnxOnlineCtcFstDecoderConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOnlineCtcFstDecoderConfig(Graph := %s, MaxActive := %d)',
  [Self.Graph, Self.MaxActive]);
end;

function TSherpaOnnxOnlineRecognizerConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOnlineRecognizerConfig(FeatConfg := %s, ' +
    'ModelConfig := %s, ' +
    'DecodingMethod := %s, ' +
    'MaxActivePaths := %d, ' +
    'EnableEndpoint := %s, ' +
    'Rule1MinTrailingSilence := %.1f, ' +
    'Rule2MinTrailingSilence := %.1f, ' +
    'Rule3MinUtteranceLength := %.1f, ' +
    'HotwordsFile := %s, ' +
    'HotwordsScore := %.1f, ' +
    'CtcFstDecoderConfig := %s, ' +
    'RuleFsts := %s, ' +
    'RuleFars := %s, ' +
    'BlankPenalty := %.1f' +
    ')'
    ,
    [Self.FeatConfig.ToString, Self.ModelConfig.ToString,
     Self.DecodingMethod, Self.MaxActivePaths, Self.EnableEndpoint.ToString,
     Self.Rule1MinTrailingSilence, Self.Rule2MinTrailingSilence,
     Self.Rule3MinUtteranceLength, Self.HotwordsFile, Self.HotwordsScore,
     Self.CtcFstDecoderConfig.ToString, Self.RuleFsts, Self.RuleFars,
     Self.BlankPenalty
    ]);
end;

constructor TSherpaOnnxOnlineRecognizer.Create(Config: TSherpaOnnxOnlineRecognizerConfig);
var
  C: SherpaOnnxOnlineRecognizerConfig;
begin
  Initialize(C);

  C.FeatConfig.SampleRate := Config.FeatConfig.SampleRate;
  C.FeatConfig.FeatureDim := Config.FeatConfig.FeatureDim;

  C.ModelConfig.Transducer.Encoder := PAnsiChar(Config.ModelConfig.Transducer.Encoder);
  C.ModelConfig.Transducer.Decoder := PAnsiChar(Config.ModelConfig.Transducer.Decoder);
  C.ModelConfig.Transducer.Joiner := PAnsiChar(Config.ModelConfig.Transducer.Joiner);

  C.ModelConfig.Paraformer.Encoder := PAnsiChar(Config.ModelConfig.Paraformer.Encoder);
  C.ModelConfig.Paraformer.Decoder := PAnsiChar(Config.ModelConfig.Paraformer.Decoder);

  C.ModelConfig.Zipformer2Ctc.Model := PAnsiChar(Config.ModelConfig.Zipformer2Ctc.Model);

  C.ModelConfig.Tokens := PAnsiChar(Config.ModelConfig.Tokens);
  C.ModelConfig.NumThreads := Config.ModelConfig.NumThreads;
  C.ModelConfig.Provider := PAnsiChar(Config.ModelConfig.Provider);
  C.ModelConfig.Debug := Ord(Config.ModelConfig.Debug);
  C.ModelConfig.ModelType := PAnsiChar(Config.ModelConfig.ModelType);
  C.ModelConfig.ModelingUnit := PAnsiChar(Config.ModelConfig.ModelingUnit);
  C.ModelConfig.BpeVocab := PAnsiChar(Config.ModelConfig.BpeVocab);

  C.DecodingMethod := PAnsiChar(Config.DecodingMethod);
  C.MaxActivePaths := Config.MaxActivePaths;
  C.EnableEndpoint := Ord(Config.EnableEndpoint);
  C.Rule1MinTrailingSilence := Config.Rule1MinTrailingSilence;
  C.Rule2MinTrailingSilence := Config.Rule2MinTrailingSilence;
  C.Rule3MinUtteranceLength := Config.Rule3MinUtteranceLength;
  C.HotwordsFile := PAnsiChar(Config.HotwordsFile);
  C.HotwordsScore := Config.HotwordsScore;
  C.CtcFstDecoderConfig.Graph := PAnsiChar(Config.CtcFstDecoderConfig.Graph);
  C.CtcFstDecoderConfig.MaxActive := Config.CtcFstDecoderConfig.MaxActive;
  C.RuleFsts := PAnsiChar(Config.RuleFsts);
  C.RuleFars := PAnsiChar(Config.RuleFars);
  C.BlankPenalty := Config.BlankPenalty;

  Self.handle := SherpaOnnxCreateOnlineRecognizerWrapper(@C);
end;

destructor TSherpaOnnxOnlineRecognizer.Destroy;
begin
  SherpaOnnxDestroyOnlineRecognizer(Self.handle);
  Self.handle := nil;
end;

end.
