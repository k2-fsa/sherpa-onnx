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
    Tokens: array of AnsiString;
    Timestamps: array of Single;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOnlineStream = class
  private
   Handle: Pointer;
  public
    constructor Create(P: Pointer);
    destructor Destroy; override;
    procedure AcceptWaveform(Samples: array of Single; SampleRate: Integer);
    procedure InputFinished;
  end;

  TSherpaOnnxOnlineRecognizer = class
  private
   Handle: Pointer;
  public
    constructor Create(Config: TSherpaOnnxOnlineRecognizerConfig);
    destructor Destroy; override;

    function CreateStream: TSherpaOnnxOnlineStream; overload;
    function CreateStream(Hotwords: AnsiString): TSherpaOnnxOnlineStream; overload;
    function IsReady(Stream: TSherpaOnnxOnlineStream): Boolean;
    procedure Decode(Stream: TSherpaOnnxOnlineStream);
    procedure Reset(Stream: TSherpaOnnxOnlineStream);
    function IsEndpoint(Stream: TSherpaOnnxOnlineStream): Boolean;
    function GetResult(Stream: TSherpaOnnxOnlineStream): TSherpaOnnxOnlineRecognizerResult;
  end;

  TSherpaOnnxOfflineTransducerModelConfig = record
    Encoder: AnsiString;
    Decoder: AnsiString;
    Joiner: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineParaformerModelConfig = record
    Model: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineNemoEncDecCtcModelConfig = record
    Model: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineWhisperModelConfig = record
    Encoder: AnsiString;
    Decoder: AnsiString;
    Language: AnsiString;
    Task: AnsiString;
    TailPaddings: Integer;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineTdnnModelConfig = record
    Model: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineLMConfig = record
    Model: AnsiString;
    Scale: Single;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineSenseVoiceModelConfig = record
    Model: AnsiString;
    Language: AnsiString;
    UseItn: Boolean;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineModelConfig = record
    Transducer: TSherpaOnnxOfflineTransducerModelConfig;
    Paraformer: TSherpaOnnxOfflineParaformerModelConfig;
    NeMoCtc: TSherpaOnnxOfflineNemoEncDecCtcModelConfig;
    Whisper: TSherpaOnnxOfflineWhisperModelConfig;
    Tdnn: TSherpaOnnxOfflineTdnnModelConfig;
    Tokens: AnsiString;
    NumThreads: Integer;
    Debug: Boolean;
    Provider: AnsiString;
    ModelType: AnsiString;
    ModelingUnit: AnsiString;
    BpeVocab: AnsiString;
    TeleSpeechCtc: AnsiString;
    SenseVoice: TSherpaOnnxOfflineSenseVoiceModelConfig;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineRecognizerConfig = record
    FeatConfig: TSherpaOnnxFeatureConfig;
    ModelConfig: TSherpaOnnxOfflineModelConfig;
    LMConfig: TSherpaOnnxOfflineLMConfig;
    DecodingMethod: AnsiString;
    MaxActivePaths: Integer;
    HotwordsFile: AnsiString;
    HotwordsScore: Single;
    RuleFsts: AnsiString;
    RuleFars: AnsiString;
    BlankPenalty: Single;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineRecognizerResult = record
    Text: AnsiString;
    Tokens: array of AnsiString;
    Timestamps: array of Single;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineStream = class
  private
   Handle: Pointer;
  public
    constructor Create(P: Pointer);
    destructor Destroy; override;
    procedure AcceptWaveform(Samples: array of Single; SampleRate: Integer);
  end;

  TSherpaOnnxOfflineRecognizer = class
  private
   Handle: Pointer;
  public
    constructor Create(Config: TSherpaOnnxOfflineRecognizerConfig);
    destructor Destroy; override;
    function CreateStream: TSherpaOnnxOfflineStream;
    procedure Decode(Stream: TSherpaOnnxOfflineStream);
    function GetResult(Stream: TSherpaOnnxOfflineStream): TSherpaOnnxOfflineRecognizerResult;
  end;

{ It supports reading a single channel wave with 16-bit encoded samples.
  Samples are normalized to the range [-1, 1].
}
function SherpaOnnxReadWave(Filename: AnsiString): TSherpaOnnxWave;

implementation

uses
  ctypes,
  fpjson,
    { See
      - https://wiki.freepascal.org/fcl-json
      - https://www.freepascal.org/daily/doc/fcl/fpjson/getjson.html
    }
  jsonparser,
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

  SherpaOnnxOfflineTransducerModelConfig = record
    Encoder: PAnsiChar;
    Decoder: PAnsiChar;
    Joiner: PAnsiChar;
  end;
  SherpaOnnxOfflineParaformerModelConfig = record
    Model: PAnsiChar;
  end;
  SherpaOnnxOfflineNemoEncDecCtcModelConfig = record
    Model: PAnsiChar;
  end;
  SherpaOnnxOfflineWhisperModelConfig = record
    Encoder: PAnsiChar;
    Decoder: PAnsiChar;
    Language: PAnsiChar;
    Task: PAnsiChar;
    TailPaddings: cint32;
  end;
  SherpaOnnxOfflineTdnnModelConfig = record
    Model: PAnsiChar;
  end;
  SherpaOnnxOfflineLMConfig = record
    Model: PAnsiChar;
    Scale: Single;
  end;
  SherpaOnnxOfflineSenseVoiceModelConfig = record
    Model: PAnsiChar;
    Language: PAnsiChar;
    UseItn: cint32;
  end;
  SherpaOnnxOfflineModelConfig = record
    Transducer: SherpaOnnxOfflineTransducerModelConfig;
    Paraformer: SherpaOnnxOfflineParaformerModelConfig;
    NeMoCtc: SherpaOnnxOfflineNemoEncDecCtcModelConfig;
    Whisper: SherpaOnnxOfflineWhisperModelConfig;
    Tdnn: SherpaOnnxOfflineTdnnModelConfig;
    Tokens: PAnsiChar;
    NumThreads: cint32;
    Debug: cint32;
    Provider: PAnsiChar;
    ModelType: PAnsiChar;
    ModelingUnit: PAnsiChar;
    BpeVocab: PAnsiChar;
    TeleSpeechCtc: PAnsiChar;
    SenseVoice:  SherpaOnnxOfflineSenseVoiceModelConfig;
  end;

  SherpaOnnxOfflineRecognizerConfig = record
    FeatConfig: SherpaOnnxFeatureConfig;
    ModelConfig: SherpaOnnxOfflineModelConfig;
    LMConfig: SherpaOnnxOfflineLMConfig;
    DecodingMethod: PAnsiChar;
    MaxActivePaths: cint32;
    HotwordsFile: PAnsiChar;
    HotwordsScore: Single;
    RuleFsts: PAnsiChar;
    RuleFars: PAnsiChar;
    BlankPenalty: Single;
  end;

  PSherpaOnnxOfflineRecognizerConfig = ^SherpaOnnxOfflineRecognizerConfig;

function SherpaOnnxCreateOnlineRecognizer(Config: PSherpaOnnxOnlineRecognizerConfig): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOnlineRecognizer(Recognizer: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateOnlineStream(Recognizer: Pointer): Pointer; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateOnlineStreamWithHotwords(Recognizer: Pointer; Hotwords: PAnsiChar): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOnlineStream(Recognizer: Pointer); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxOnlineStreamAcceptWaveform(Stream: Pointer;
  SampleRate: cint32; Samples: pcfloat; N: cint32 ); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxOnlineStreamInputFinished(Stream: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxIsOnlineStreamReady(Recognizer: Pointer; Stream: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDecodeOnlineStream(Recognizer: Pointer; Stream: Pointer); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxOnlineStreamReset(Recognizer: Pointer; Stream: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOnlineStreamIsEndpoint(Recognizer: Pointer; Stream: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxGetOnlineStreamResultAsJson(Recognizer: Pointer; Stream: Pointer): PAnsiChar; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOnlineStreamResultJson(PJson: PAnsiChar); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateOfflineRecognizer(Config: PSherpaOnnxOfflineRecognizerConfig): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOfflineRecognizer(Recognizer: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateOfflineStream(Recognizer: Pointer): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOfflineStream(Stream: Pointer); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxAcceptWaveformOffline(Stream: Pointer;
  SampleRate: cint32; Samples: pcfloat; N: cint32); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDecodeOfflineStream(Recognizer: Pointer; Stream: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxGetOfflineStreamResultAsJson(Stream: Pointer): PAnsiChar; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOfflineStreamResultJson(Json: PAnsiChar); cdecl;
  external SherpaOnnxLibName;

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

  Result.Samples := nil;
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
  Result := Format('TSherpaOnnxOnlineRecognizerConfig(FeatConfig := %s, ' +
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

function TSherpaOnnxOnlineRecognizerResult.ToString: AnsiString;
var
  TokensStr: AnsiString;
  S: AnsiString;
  TimestampStr: AnsiString;
  T: Single;
  Sep: AnsiString;
begin
  TokensStr := '[';
  Sep := '';
  for S in Self.Tokens do
  begin
    TokensStr := TokensStr + Sep + S;
    Sep := ', ';
  end;
  TokensStr := TokensStr + ']';

  TimestampStr := '[';
  Sep := '';
  for T in Self.Timestamps do
  begin
    TimestampStr := TimestampStr + Sep + Format('%.2f', [T]);
    Sep := ', ';
  end;
  TimestampStr := TimestampStr + ']';

  Result := Format('TSherpaOnnxOnlineRecognizerResult(Text := %s, ' +
    'Tokens := %s, ' +
    'Timestamps := %s' +
    ')',
    [Self.Text, TokensStr, TimestampStr]);
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

  Self.Handle := SherpaOnnxCreateOnlineRecognizer(@C);
end;

destructor TSherpaOnnxOnlineRecognizer.Destroy;
begin
  SherpaOnnxDestroyOnlineRecognizer(Self.Handle);
  Self.Handle := nil;
end;

function TSherpaOnnxOnlineRecognizer.CreateStream: TSherpaOnnxOnlineStream;
var
  Stream: Pointer;
begin
  Stream := SherpaOnnxCreateOnlineStream(Self.Handle);
  Result := TSherpaOnnxOnlineStream.Create(Stream);
end;

function TSherpaOnnxOnlineRecognizer.CreateStream(Hotwords: AnsiString): TSherpaOnnxOnlineStream;
var
  Stream: Pointer;
begin
  Stream := SherpaOnnxCreateOnlineStreamWithHotwords(Self.Handle, PAnsiChar(Hotwords));
  Result := TSherpaOnnxOnlineStream.Create(Stream);
end;

function TSherpaOnnxOnlineRecognizer.IsReady(Stream: TSherpaOnnxOnlineStream): Boolean;
begin
  Result := SherpaOnnxIsOnlineStreamReady(Self.Handle, Stream.Handle) = 1;
end;

procedure TSherpaOnnxOnlineRecognizer.Decode(Stream: TSherpaOnnxOnlineStream);
begin
  SherpaOnnxDecodeOnlineStream(Self.Handle, Stream.Handle);
end;

procedure TSherpaOnnxOnlineRecognizer.Reset(Stream: TSherpaOnnxOnlineStream);
begin
  SherpaOnnxOnlineStreamReset(Self.Handle, Stream.Handle);
end;

function TSherpaOnnxOnlineRecognizer.IsEndpoint(Stream: TSherpaOnnxOnlineStream): Boolean;
begin
  Result := SherpaOnnxOnlineStreamIsEndpoint(Self.Handle, Stream.Handle) = 1;
end;

function TSherpaOnnxOnlineRecognizer.GetResult(Stream: TSherpaOnnxOnlineStream): TSherpaOnnxOnlineRecognizerResult;
var
  pJson: PAnsiChar;
  JsonData: TJSONData;
  JsonObject : TJSONObject;
  JsonEnum: TJSONEnum;
  I: Integer;
begin
  pJson := SherpaOnnxGetOnlineStreamResultAsJson(Self.Handle, Stream.Handle);

  {
   - https://www.freepascal.org/daily/doc/fcl/fpjson/getjson.html
   - https://www.freepascal.org/daily/doc/fcl/fpjson/tjsondata.html
   - https://www.freepascal.org/daily/doc/fcl/fpjson/tjsonobject.html
   - https://www.freepascal.org/daily/doc/fcl/fpjson/tjsonenum.html
  }

  JsonData := GetJSON(AnsiString(pJson), False);

  JsonObject := JsonData as TJSONObject;

  Result.Text := JsonObject.Strings['text'];

  SetLength(Result.Tokens, JsonObject.Arrays['tokens'].Count);

  I := 0;
  for JsonEnum in JsonObject.Arrays['tokens'] do
  begin
    Result.Tokens[I] := JsonEnum.Value.AsString;
    Inc(I);
  end;

  SetLength(Result.Timestamps, JsonObject.Arrays['timestamps'].Count);
  I := 0;
  for JsonEnum in JsonObject.Arrays['timestamps'] do
  begin
    Result.Timestamps[I] := JsonEnum.Value.AsFloat;
    Inc(I);
  end;

  SherpaOnnxDestroyOnlineStreamResultJson(pJson);
end;


constructor TSherpaOnnxOnlineStream.Create(P: Pointer);
begin
  Self.Handle := P;
end;

destructor TSherpaOnnxOnlineStream.Destroy;
begin
  SherpaOnnxDestroyOnlineStream(Self.Handle);
  Self.Handle := nil;
end;

procedure TSherpaOnnxOnlineStream.AcceptWaveform(Samples: array of Single; SampleRate: Integer);
begin
  SherpaOnnxOnlineStreamAcceptWaveform(Self.Handle, SampleRate,
    pcfloat(Samples), Length(Samples));
end;

procedure TSherpaOnnxOnlineStream.InputFinished;
begin
  SherpaOnnxOnlineStreamInputFinished(Self.Handle);
end;

function TSherpaOnnxOfflineTransducerModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineTransducerModelConfig(' +
    'Encoder := %s, ' +
    'Decoder := %s, ' +
    'Joiner := %s' +
    ')',
    [Self.Encoder, Self.Decoder, Self.Joiner]);
end;

function TSherpaOnnxOfflineParaformerModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineParaformerModelConfig(Model := %s)',
    [Self.Model]);
end;

function TSherpaOnnxOfflineNemoEncDecCtcModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineNemoEncDecCtcModelConfig(Model := %s)',
    [Self.Model]);
end;

function TSherpaOnnxOfflineWhisperModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineWhisperModelConfig(' +
    'Encoder := %s, ' +
    'Decoder := %s, ' +
    'Language := %s, ' +
    'Task := %s, ' +
    'TailPaddings := %d' +
    ')',
    [Self.Encoder, Self.Decoder, Self.Language, Self.Task, Self.TailPaddings]);
end;

function TSherpaOnnxOfflineTdnnModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineTdnnModelConfig(Model := %s)',
    [Self.Model]);
end;

function TSherpaOnnxOfflineLMConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineLMConfig(' +
    'Model := %s, ' +
    'Scale := %.1f' +
    ')',
    [Self.Model, Self.Scale]);
end;

function TSherpaOnnxOfflineSenseVoiceModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineSenseVoiceModelConfig(' +
    'Model := %s, ' +
    'Language := %s, ' +
    'UseItn := %s' +
    ')',
    [Self.Model, Self.Language, Self.UseItn.ToString]);
end;

function TSherpaOnnxOfflineModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineModelConfig(' +
    'Transducer := %s, ' +
    'Paraformer := %s, ' +
    'NeMoCtc := %s, ' +
    'Whisper := %s, ' +
    'Tdnn := %s, ' +
    'Tokens := %s, ' +
    'NumThreads := %d, ' +
    'Debug := %s, ' +
    'Provider := %s, ' +
    'ModelType := %s, ' +
    'ModelingUnit := %s, ' +
    'BpeVocab := %s, ' +
    'TeleSpeechCtc := %s, ' +
    'SenseVoice := %s' +
    ')',
    [Self.Transducer.ToString, Self.Paraformer.ToString,
     Self.NeMoCtc.ToString, Self.Whisper.ToString, Self.Tdnn.ToString,
     Self.Tokens, Self.NumThreads, Self.Debug.ToString, Self.Provider,
     Self.ModelType, Self.ModelingUnit, Self.BpeVocab,
     Self.TeleSpeechCtc, Self.SenseVoice.ToString
     ]);
end;

function TSherpaOnnxOfflineRecognizerConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineRecognizerConfig(' +
    'FeatConfig := %s, ' +
    'ModelConfig := %s, ' +
    'LMConfig := %s, ' +
    'DecodingMethod := %s, ' +
    'MaxActivePaths := %d, ' +
    'HotwordsFile := %s, ' +
    'HotwordsScore := %.1f, ' +
    'RuleFsts := %s, ' +
    'RuleFars := %s, ' +
    'BlankPenalty := %1.f' +
    ')',
    [Self.FeatConfig.ToString, Self.ModelConfig.ToString,
     Self.LMConfig.ToString, Self.DecodingMethod, Self.MaxActivePaths,
     Self.HotwordsFile, Self.HotwordsScore, Self.RuleFsts, Self.RuleFars,
     Self.BlankPenalty
     ]);
end;

constructor TSherpaOnnxOfflineRecognizer.Create(Config: TSherpaOnnxOfflineRecognizerConfig);
var
  C: SherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(C);

  C.FeatConfig.SampleRate := Config.FeatConfig.SampleRate;
  C.FeatConfig.FeatureDim := Config.FeatConfig.FeatureDim;

  C.ModelConfig.Transducer.Encoder := PAnsiChar(Config.ModelConfig.Transducer.Encoder);
  C.ModelConfig.Transducer.Decoder := PAnsiChar(Config.ModelConfig.Transducer.Decoder);
  C.ModelConfig.Transducer.Joiner := PAnsiChar(Config.ModelConfig.Transducer.Joiner);

  C.ModelConfig.Paraformer.Model := PAnsiChar(Config.ModelConfig.Paraformer.Model);
  C.ModelConfig.NeMoCtc.Model := PAnsiChar(Config.ModelConfig.NeMoCtc.Model);

  C.ModelConfig.Whisper.Encoder := PAnsiChar(Config.ModelConfig.Whisper.Encoder);
  C.ModelConfig.Whisper.Decoder := PAnsiChar(Config.ModelConfig.Whisper.Decoder);
  C.ModelConfig.Whisper.Language := PAnsiChar(Config.ModelConfig.Whisper.Language);
  C.ModelConfig.Whisper.Task := PAnsiChar(Config.ModelConfig.Whisper.Task);
  C.ModelConfig.Whisper.TailPaddings := Config.ModelConfig.Whisper.TailPaddings;

  C.ModelConfig.Tdnn.Model := PAnsiChar(Config.ModelConfig.Tdnn.Model);


  C.ModelConfig.Tokens := PAnsiChar(Config.ModelConfig.Tokens);
  C.ModelConfig.NumThreads := Config.ModelConfig.NumThreads;
  C.ModelConfig.Debug := Ord(Config.ModelConfig.Debug);
  C.ModelConfig.Provider := PAnsiChar(Config.ModelConfig.Provider);
  C.ModelConfig.ModelType := PAnsiChar(Config.ModelConfig.ModelType);
  C.ModelConfig.ModelingUnit := PAnsiChar(Config.ModelConfig.ModelingUnit);
  C.ModelConfig.BpeVocab := PAnsiChar(Config.ModelConfig.BpeVocab);
  C.ModelConfig.TeleSpeechCtc := PAnsiChar(Config.ModelConfig.TeleSpeechCtc);

  C.ModelConfig.SenseVoice.Model := PAnsiChar(Config.ModelConfig.SenseVoice.Model);
  C.ModelConfig.SenseVoice.Language := PAnsiChar(Config.ModelConfig.SenseVoice.Language);
  C.ModelConfig.SenseVoice.UseItn := Ord(Config.ModelConfig.SenseVoice.UseItn);

  C.LMConfig.Model := PAnsiChar(Config.LMConfig.Model);
  C.LMConfig.Scale := Config.LMConfig.Scale;

  C.DecodingMethod := PAnsiChar(Config.DecodingMethod);
  C.MaxActivePaths := Config.MaxActivePaths;
  C.HotwordsFile := PAnsiChar(Config.HotwordsFile);
  C.HotwordsScore := Config.HotwordsScore;
  C.RuleFsts := PAnsiChar(Config.RuleFsts);
  C.RuleFars := PAnsiChar(Config.RuleFars);
  C.BlankPenalty := Config.BlankPenalty;

  Self.Handle := SherpaOnnxCreateOfflineRecognizer(@C);
end;

destructor TSherpaOnnxOfflineRecognizer.Destroy;
begin
  SherpaOnnxDestroyOfflineRecognizer(Self.Handle);
  Self.Handle := nil;
end;

function TSherpaOnnxOfflineRecognizer.CreateStream: TSherpaOnnxOfflineStream;
var
  Stream: Pointer;
begin
  Stream := SherpaOnnxCreateOfflineStream(Self.Handle);
  Result := TSherpaOnnxOfflineStream.Create(Stream);
end;

procedure TSherpaOnnxOfflineRecognizer.Decode(Stream: TSherpaOnnxOfflineStream);
begin
  SherpaOnnxDecodeOfflineStream(Self.Handle, Stream.Handle);
end;

function TSherpaOnnxOfflineRecognizer.GetResult(Stream: TSherpaOnnxOfflineStream): TSherpaOnnxOfflineRecognizerResult;
var
  pJson: PAnsiChar;
  JsonData: TJSONData;
  JsonObject : TJSONObject;
  JsonEnum: TJSONEnum;
  I: Integer;
begin
  pJson := SherpaOnnxGetOfflineStreamResultAsJson(Stream.Handle);

  JsonData := GetJSON(AnsiString(pJson), False);

  JsonObject := JsonData as TJSONObject;

  Result.Text := JsonObject.Strings['text'];

  SetLength(Result.Tokens, JsonObject.Arrays['tokens'].Count);

  I := 0;
  for JsonEnum in JsonObject.Arrays['tokens'] do
  begin
    Result.Tokens[I] := JsonEnum.Value.AsString;
    Inc(I);
  end;

  SetLength(Result.Timestamps, JsonObject.Arrays['timestamps'].Count);
  I := 0;
  for JsonEnum in JsonObject.Arrays['timestamps'] do
  begin
    Result.Timestamps[I] := JsonEnum.Value.AsFloat;
    Inc(I);
  end;

  SherpaOnnxDestroyOfflineStreamResultJson(pJson);
end;

constructor TSherpaOnnxOfflineStream.Create(P: Pointer);
begin
  Self.Handle := P;
end;

destructor TSherpaOnnxOfflineStream.Destroy;
begin
  SherpaOnnxDestroyOfflineStream(Self.Handle);
  Self.Handle := nil;
end;

procedure TSherpaOnnxOfflineStream.AcceptWaveform(Samples: array of Single; SampleRate: Integer);
begin
  SherpaOnnxAcceptWaveformOffline(Self.Handle, SampleRate, pcfloat(Samples),
    Length(Samples));
end;

function TSherpaOnnxOfflineRecognizerResult.ToString: AnsiString;
var
  TokensStr: AnsiString;
  S: AnsiString;
  TimestampStr: AnsiString;
  T: Single;
  Sep: AnsiString;
begin
  TokensStr := '[';
  Sep := '';
  for S in Self.Tokens do
  begin
    TokensStr := TokensStr + Sep + S;
    Sep := ', ';
  end;
  TokensStr := TokensStr + ']';

  TimestampStr := '[';
  Sep := '';
  for T in Self.Timestamps do
  begin
    TimestampStr := TimestampStr + Sep + Format('%.2f', [T]);
    Sep := ', ';
  end;
  TimestampStr := TimestampStr + ']';

  Result := Format('TSherpaOnnxOfflineRecognizerResult(Text := %s, ' +
    'Tokens := %s, ' +
    'Timestamps := %s' +
    ')',
    [Self.Text, TokensStr, TimestampStr]);
end;

end.

