{ Copyright (c)  2024  Xiaomi Corporation

Please see
https://github.com/k2-fsa/sherpa-onnx/tree/master/pascal-api-examples
for how to use APIs in this file.
}

unit sherpa_onnx;

{$IFDEF FPC}
  {$mode objfpc}
  {$modeSwitch advancedRecords} { to support records with methods }
{$ENDIF}

{$LongStrings ON}

interface
uses
  ctypes;

type
  TSherpaOnnxSamplesArray = array of Single;

  TSherpaOnnxLinearResampler = class
  private
    Handle: Pointer;
    InputSampleRate: Integer;
    OutputSampleRate: Integer;
  public
    constructor Create(SampleRateIn: Integer; SampleRateOut: Integer);
    destructor Destroy; override;

    function Resample(Samples: pcfloat;
      N: Integer; Flush: Boolean): TSherpaOnnxSamplesArray; overload;

    function Resample(Samples: array of Single;
      Flush: Boolean): TSherpaOnnxSamplesArray; overload;

    procedure Reset;

    property GetInputSampleRate: Integer Read InputSampleRate;
    property GetOutputSampleRate: Integer Read OutputSampleRate;
  end;

  PSherpaOnnxGeneratedAudioCallbackWithArg = ^TSherpaOnnxGeneratedAudioCallbackWithArg;

  TSherpaOnnxGeneratedAudioCallbackWithArg = function(
      Samples: pcfloat; N: cint32;
      Arg: Pointer): cint; cdecl;

  TSherpaOnnxOfflineTtsVitsModelConfig = record
    Model: AnsiString;
    Lexicon: AnsiString;
    Tokens: AnsiString;
    DataDir: AnsiString;
    NoiseScale: Single;
    NoiseScaleW: Single;
    LengthScale: Single;
    DictDir: AnsiString;

    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineTtsVitsModelConfig);
  end;

  TSherpaOnnxOfflineTtsModelConfig = record
    Vits: TSherpaOnnxOfflineTtsVitsModelConfig;
    NumThreads: Integer;
    Debug: Boolean;
    Provider: AnsiString;

    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineTtsModelConfig);
  end;

  TSherpaOnnxOfflineTtsConfig = record
    Model: TSherpaOnnxOfflineTtsModelConfig;
    RuleFsts: AnsiString;
    MaxNumSentences: Integer;
    RuleFars: AnsiString;

    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineTtsConfig);
  end;

  TSherpaOnnxGeneratedAudio = record
    Samples: array of Single;
    SampleRate: Integer;
  end;

  TSherpaOnnxOfflineTts = class
  private
   Handle: Pointer;
   SampleRate: Integer;
   NumSpeakers: Integer;
   _Config: TSherpaOnnxOfflineTtsConfig;
  public
    constructor Create(Config: TSherpaOnnxOfflineTtsConfig);
    destructor Destroy; override;

    function Generate(Text: AnsiString; SpeakerId: Integer;
      Speed: Single): TSherpaOnnxGeneratedAudio; overload;

    function Generate(Text: AnsiString; SpeakerId: Integer;
      Speed: Single;
      Callback: PSherpaOnnxGeneratedAudioCallbackWithArg;
      Arg: Pointer
      ): TSherpaOnnxGeneratedAudio; overload;

    property GetHandle: Pointer Read Handle;
    property GetSampleRate: Integer Read SampleRate;
    property GetNumSpeakers: Integer Read NumSpeakers;
  end;

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
    TokensBuf: AnsiString;
    TokensBufSize: Integer;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOnlineModelConfig);
  end;

  TSherpaOnnxFeatureConfig = record
    SampleRate: Integer;
    FeatureDim: Integer;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxFeatureConfig);
  end;

  TSherpaOnnxOnlineCtcFstDecoderConfig = record
    Graph: AnsiString;
    MaxActive: Integer;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOnlineCtcFstDecoderConfig);
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
    HotwordsBuf: AnsiString;
    HotwordsBufSize: Integer;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOnlineRecognizerConfig);
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
    property GetHandle: Pointer Read Handle;
  end;

  TSherpaOnnxOnlineRecognizer = class
  private
   Handle: Pointer;
   _Config: TSherpaOnnxOnlineRecognizerConfig;
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
    property Config: TSherpaOnnxOnlineRecognizerConfig Read _Config;
    property GetHandle: Pointer Read Handle;
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
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineWhisperModelConfig);
  end;

  TSherpaOnnxOfflineTdnnModelConfig = record
    Model: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineLMConfig = record
    Model: AnsiString;
    Scale: Single;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineLMConfig);
  end;

  TSherpaOnnxOfflineSenseVoiceModelConfig = record
    Model: AnsiString;
    Language: AnsiString;
    UseItn: Boolean;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineSenseVoiceModelConfig);
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
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineModelConfig);
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
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineRecognizerConfig);
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
    property GetHandle: Pointer Read Handle;
  end;

  TSherpaOnnxOfflineRecognizer = class
  private
   Handle: Pointer;
   _Config: TSherpaOnnxOfflineRecognizerConfig;
  public
    constructor Create(Config: TSherpaOnnxOfflineRecognizerConfig);
    destructor Destroy; override;
    function CreateStream: TSherpaOnnxOfflineStream;
    procedure Decode(Stream: TSherpaOnnxOfflineStream);
    function GetResult(Stream: TSherpaOnnxOfflineStream): TSherpaOnnxOfflineRecognizerResult;
    property Config: TSherpaOnnxOfflineRecognizerConfig Read _Config;
    property GetHandle: Pointer Read Handle;
  end;

  TSherpaOnnxSileroVadModelConfig = record
    Model: AnsiString;
    Threshold: Single;
    MinSilenceDuration: Single;
    MinSpeechDuration: Single;
    WindowSize: Integer;
    MaxSpeechDuration: Single;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxSileroVadModelConfig);
  end;

  TSherpaOnnxVadModelConfig = record
    SileroVad: TSherpaOnnxSileroVadModelConfig;
    SampleRate: Integer;
    NumThreads: Integer;
    Provider: AnsiString;
    Debug: Boolean;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxVadModelConfig);
  end;


  TSherpaOnnxCircularBuffer = class
  private
    Handle: Pointer;
  public
    constructor Create(Capacity: Integer);
    destructor Destroy; override;
    procedure Push(Samples: array of Single); overload;
    procedure Push(Samples: pcfloat; N: Integer); overload;
    function Get(StartIndex: Integer; N: Integer): TSherpaOnnxSamplesArray;
    procedure Pop(N: Integer);
    procedure Reset;
    function Size: Integer;
    function Head: Integer;
    property GetHandle: Pointer Read Handle;
  end;

  TSherpaOnnxSpeechSegment = record
    Samples: array of Single;
    Start: Integer;
  end;

  TSherpaOnnxVoiceActivityDetector = class
  private
    Handle: Pointer;
    _Config: TSherpaOnnxVadModelConfig;
  public
    constructor Create(Config: TSherpaOnnxVadModelConfig; BufferSizeInSeconds: Single);
    destructor Destroy; override;
    procedure AcceptWaveform(Samples: array of Single); overload;
    procedure AcceptWaveform(Samples: array of Single; Offset: Integer; N: Integer); overload;
    function IsEmpty: Boolean;
    function IsDetected: Boolean;
    procedure Pop;
    procedure Clear;
    function Front: TSherpaOnnxSpeechSegment;
    procedure Reset;
    procedure Flush;
    property Config: TSherpaOnnxVadModelConfig Read _Config;
    property GetHandle: Pointer Read Handle;
  end;


  TSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig = record
    Model: AnsiString;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineSpeakerSegmentationModelConfig = record
    Pyannote: TSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig;
    NumThreads: Integer;
    Debug: Boolean;
    Provider: AnsiString;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineSpeakerSegmentationModelConfig);
  end;

  TSherpaOnnxFastClusteringConfig = record
    NumClusters: Integer;
    Threshold: Single;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxFastClusteringConfig);
  end;

  TSherpaOnnxSpeakerEmbeddingExtractorConfig = record
    Model: AnsiString;
    NumThreads: Integer;
    Debug: Boolean;
    Provider: AnsiString;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxSpeakerEmbeddingExtractorConfig);
  end;

  TSherpaOnnxOfflineSpeakerDiarizationConfig = record
    Segmentation: TSherpaOnnxOfflineSpeakerSegmentationModelConfig;
    Embedding: TSherpaOnnxSpeakerEmbeddingExtractorConfig;
    Clustering: TSherpaOnnxFastClusteringConfig;
    MinDurationOn: Single;
    MinDurationOff: Single;
    function ToString: AnsiString;
    class operator Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineSpeakerDiarizationConfig);
  end;

  TSherpaOnnxOfflineSpeakerDiarizationSegment = record
    Start: Single;
    Stop: Single;
    Speaker: Integer;
    function ToString: AnsiString;
  end;

  TSherpaOnnxOfflineSpeakerDiarizationSegmentArray = array of TSherpaOnnxOfflineSpeakerDiarizationSegment;

  PSherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg = ^TSherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg;

  TSherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg = function(
      NumProcessChunks: cint32;
      NumTotalChunks: cint32): cint32; cdecl;

  TSherpaOnnxOfflineSpeakerDiarization = class
  private
    Handle: Pointer;
    SampleRate: Integer;
    _Config: TSherpaOnnxOfflineSpeakerDiarizationConfig;
  public
    constructor Create(Config: TSherpaOnnxOfflineSpeakerDiarizationConfig);
    destructor Destroy; override;
    procedure SetConfig(Config: TSherpaOnnxOfflineSpeakerDiarizationConfig);
    function Process(Samples: array of Single): TSherpaOnnxOfflineSpeakerDiarizationSegmentArray; overload;
    function Process(Samples: array of Single; Callback: PSherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg): TSherpaOnnxOfflineSpeakerDiarizationSegmentArray; overload;
    property GetHandle: Pointer Read Handle;
    property GetSampleRate: Integer Read SampleRate;
  end;


  { It supports reading a single channel wave with 16-bit encoded samples.
    Samples are normalized to the range [-1, 1].
  }
  function SherpaOnnxReadWave(Filename: AnsiString): TSherpaOnnxWave;

  function SherpaOnnxWriteWave(Filename: AnsiString;
    Samples: array of Single; SampleRate: Integer): Boolean;

implementation

uses
  fpjson,
    { See
      - https://wiki.freepascal.org/fcl-json
      - https://www.freepascal.org/daily/doc/fcl/fpjson/getjson.html
    }
  jsonparser,
  SysUtils;

const
  {
  See
   - https://www.freepascal.org/docs-html/prog/progap7.html
   - https://downloads.freepascal.org/fpc/docs-pdf/
   - https://downloads.freepascal.org/fpc/docs-pdf/CinFreePascal.pdf
  }

  {$if defined(WINDOWS)}
   { For windows, we always use dynamic link. See
     https://forum.lazarus.freepascal.org/index.php/topic,15712.msg84781.html#msg84781
     We need to rebuild the static lib for windows using Mingw or cygwin
   }
     SherpaOnnxLibName = 'sherpa-onnx-c-api.dll';
  {$elseif not defined(SHERPA_ONNX_USE_SHARED_LIBS)}
     {static link for linux and macos}
     {$linklib sherpa-onnx-c-api}
     {$linklib sherpa-onnx-core}
     {$linklib kaldi-decoder-core}
     {$linklib sherpa-onnx-kaldifst-core}
     {$linklib sherpa-onnx-fstfar}
     {$linklib sherpa-onnx-fst}
     {$linklib kaldi-native-fbank-core}
     {$linklib piper_phonemize}
     {$linklib espeak-ng}
     {$linklib ucd}
     {$linklib onnxruntime}
     {$linklib ssentencepiece_core}

     {$ifdef LINUX}
       {$linklib m}
       {$LINKLIB stdc++}
       {$LINKLIB gcc_s}
     {$endif}

     {$ifdef DARWIN}
       {$linklib c++}
     {$endif}
     SherpaOnnxLibName = '';
  {$else}
     {dynamic link for linux and macos}
     SherpaOnnxLibName = 'sherpa-onnx-c-api';
     {$linklib sherpa-onnx-c-api}
  {$endif}

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
    TokensBuf: PAnsiChar;
    TokensBufSize: cint32;
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
    Rule1MinTrailingSilence: cfloat;
    Rule2MinTrailingSilence: cfloat;
    Rule3MinUtteranceLength: cfloat;
    HotwordsFile: PAnsiChar;
    HotwordsScore: cfloat;
    CtcFstDecoderConfig: SherpaOnnxOnlineCtcFstDecoderConfig;
    RuleFsts: PAnsiChar;
    RuleFars: PAnsiChar;
    BlankPenalty: cfloat;
    HotwordsBuf: PAnsiChar;
    HotwordsBufSize: cint32;
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
    Scale: cfloat;
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
    HotwordsScore: cfloat;
    RuleFsts: PAnsiChar;
    RuleFars: PAnsiChar;
    BlankPenalty: cfloat;
  end;

  PSherpaOnnxOfflineRecognizerConfig = ^SherpaOnnxOfflineRecognizerConfig;

  SherpaOnnxSileroVadModelConfig = record
    Model: PAnsiChar;
    Threshold: cfloat;
    MinSilenceDuration: cfloat;
    MinSpeechDuration: cfloat;
    WindowSize: cint32;
    MaxSpeechDuration: cfloat;
  end;
  SherpaOnnxVadModelConfig = record
    SileroVad: SherpaOnnxSileroVadModelConfig;
    SampleRate: cint32;
    NumThreads: cint32;
    Provider: PAnsiChar;
    Debug: cint32;
  end;
  PSherpaOnnxVadModelConfig = ^SherpaOnnxVadModelConfig;

  SherpaOnnxSpeechSegment = record
    Start: cint32;
    Samples: pcfloat;
    N: cint32;
  end;

  PSherpaOnnxSpeechSegment = ^SherpaOnnxSpeechSegment;

  SherpaOnnxOfflineTtsVitsModelConfig = record
    Model: PAnsiChar;
    Lexicon: PAnsiChar;
    Tokens: PAnsiChar;
    DataDir: PAnsiChar;
    NoiseScale: cfloat;
    NoiseScaleW: cfloat;
    LengthScale: cfloat;
    DictDir: PAnsiChar;
  end;

  SherpaOnnxOfflineTtsModelConfig = record
    Vits: SherpaOnnxOfflineTtsVitsModelConfig;
    NumThreads: cint32;
    Debug: cint32;
    Provider: PAnsiChar;
  end;

  SherpaOnnxOfflineTtsConfig = record
    Model: SherpaOnnxOfflineTtsModelConfig;
    RuleFsts: PAnsiChar;
    MaxNumSentences: cint32;
    RuleFars: PAnsiChar;
  end;

  PSherpaOnnxOfflineTtsConfig = ^SherpaOnnxOfflineTtsConfig;

  SherpaOnnxGeneratedAudio = record
    Samples: pcfloat;
    N: cint32;
    SampleRate: cint32;
  end;

  PSherpaOnnxGeneratedAudio = ^SherpaOnnxGeneratedAudio;

  SherpaOnnxResampleOut = record
    Samples: pcfloat;
    N: cint32;
  end;

  PSherpaOnnxResampleOut = ^SherpaOnnxResampleOut;

  SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig = record
    Model: PAnsiChar;
  end;

  SherpaOnnxOfflineSpeakerSegmentationModelConfig = record
    Pyannote: SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig;
    NumThreads: cint32;
    Debug: cint32;
    Provider: PAnsiChar;
  end;

  SherpaOnnxFastClusteringConfig = record
    NumClusters: cint32;
    Threshold: cfloat;
  end;

  SherpaOnnxSpeakerEmbeddingExtractorConfig = record
    Model: PAnsiChar;
    NumThreads: cint32;
    Debug: cint32;
    Provider: PAnsiChar;
  end;

  SherpaOnnxOfflineSpeakerDiarizationConfig = record
    Segmentation: SherpaOnnxOfflineSpeakerSegmentationModelConfig;
    Embedding: SherpaOnnxSpeakerEmbeddingExtractorConfig;
    Clustering: SherpaOnnxFastClusteringConfig;
    MinDurationOn: cfloat;
    MinDurationOff: cfloat;
  end;

  SherpaOnnxOfflineSpeakerDiarizationSegment = record
    Start: cfloat;
    Stop: cfloat;
    Speaker: cint32;
  end;

  PSherpaOnnxOfflineSpeakerDiarizationSegment = ^SherpaOnnxOfflineSpeakerDiarizationSegment;

  PSherpaOnnxOfflineSpeakerDiarizationConfig = ^SherpaOnnxOfflineSpeakerDiarizationConfig;

function SherpaOnnxCreateLinearResampler(SampleRateInHz: cint32;
  SampleRateOutHz: cint32;
  FilterCutoffHz: cfloat;
  NumZeros: cint32): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyLinearResampler(P: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxLinearResamplerResample(P: Pointer;
  Samples: pcfloat;
  N: Integer;
  Flush: Integer): PSherpaOnnxResampleOut; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxLinearResamplerResampleFree(P: PSherpaOnnxResampleOut); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxLinearResamplerReset(P: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateOfflineSpeakerDiarization(Config: PSherpaOnnxOfflineSpeakerDiarizationConfig): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOfflineSpeakerDiarization(P: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(P: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxOfflineSpeakerDiarizationSetConfig(P: Pointer; Config: PSherpaOnnxOfflineSpeakerDiarizationConfig); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(P: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(P: Pointer): PSherpaOnnxOfflineSpeakerDiarizationSegment; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxOfflineSpeakerDiarizationDestroySegment(P: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineSpeakerDiarizationProcess(P: Pointer; Samples: pcfloat; N: cint32): Pointer; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg(P: Pointer;
  Samples: pcfloat; N: cint32;  Callback: PSherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxOfflineSpeakerDiarizationDestroyResult(P: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateOfflineTts(Config: PSherpaOnnxOfflineTtsConfig): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOfflineTts(Tts: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineTtsSampleRate(Tts: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineTtsNumSpeakers(Tts: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineTtsGenerate(Tts: Pointer;
  Text: PAnsiChar; Sid: cint32; Speed: cfloat): PSherpaOnnxGeneratedAudio; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(Tts: Pointer;
  Text: PAnsiChar; Sid: cint32; Speed: cfloat;
  Callback: PSherpaOnnxGeneratedAudioCallbackWithArg;
  Arg: Pointer): PSherpaOnnxGeneratedAudio; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyOfflineTtsGeneratedAudio(Audio: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateVoiceActivityDetector(Config: PSherpaOnnxVadModelConfig;
  BufferSizeInSeconds: cfloat): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyVoiceActivityDetector(Vad: Pointer); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxVoiceActivityDetectorAcceptWaveform(Vad: Pointer;
  Samples: pcfloat; N: cint32); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxVoiceActivityDetectorEmpty(Vad: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxVoiceActivityDetectorDetected(Vad: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxVoiceActivityDetectorPop(Vad: Pointer); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxVoiceActivityDetectorClear(Vad: Pointer); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxVoiceActivityDetectorFront(Vad: Pointer): PSherpaOnnxSpeechSegment; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroySpeechSegment(P: PSherpaOnnxSpeechSegment); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxVoiceActivityDetectorReset(P: PSherpaOnnxSpeechSegment); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxVoiceActivityDetectorFlush(P: PSherpaOnnxSpeechSegment); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCreateCircularBuffer(Capacity: cint32): Pointer; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxDestroyCircularBuffer(Buffer: Pointer) ; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxCircularBufferPush(Buffer: Pointer; Samples: pcfloat; N: cint32); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCircularBufferGet(Buffer: Pointer; StartIndex: cint32; N: cint32): pcfloat ; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxCircularBufferFree(P: pcfloat); cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxCircularBufferPop(Buffer: Pointer; N: cint32); cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCircularBufferSize(Buffer: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

function SherpaOnnxCircularBufferHead(Buffer: Pointer): cint32; cdecl;
  external SherpaOnnxLibName;

procedure SherpaOnnxCircularBufferReset(Buffer: Pointer); cdecl;
  external SherpaOnnxLibName;

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

function SherpaOnnxWriteWaveWrapper(Samples: pcfloat; N: cint32;
  SampleRate: cint32; Filename: PAnsiChar): cint32; cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxWriteWave';

procedure SherpaOnnxFreeWaveWrapper(P: PSherpaOnnxWave); cdecl;
  external SherpaOnnxLibName name 'SherpaOnnxFreeWave';

function SherpaOnnxWriteWave(Filename: AnsiString;
    Samples: array of Single; SampleRate: Integer): Boolean;
begin
  Result := SherpaOnnxWriteWaveWrapper(pcfloat(Samples), Length(Samples),
    SampleRate, PAnsiChar(Filename)) = 1;
end;

function SherpaOnnxReadWave(Filename: AnsiString): TSherpaOnnxWave;
var
  PFilename: PAnsiChar;
  PWave: PSherpaOnnxWave;
  I: Integer;
begin
  Result.Samples := nil;
  Result.SampleRate := 0;

  PFilename := PAnsiChar(Filename);

  PWave := SherpaOnnxReadWaveWrapper(PFilename);

  if PWave = nil then
    Exit;


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
  C := Default(SherpaOnnxOnlineRecognizerConfig);
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
  Self._Config := Config;
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
  C := Default(SherpaOnnxOfflineRecognizerConfig);
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
  Self._Config := Config;
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

function TSherpaOnnxSileroVadModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxSileroVadModelConfig(' +
    'Model := %s, ' +
    'Threshold := %.2f, ' +
    'MinSilenceDuration := %.2f, ' +
    'MinSpeechDuration := %.2f, ' +
    'WindowSize := %d, ' +
    'MaxSpeechDuration := %.2f' +
    ')',
    [Self.Model, Self.Threshold, Self.MinSilenceDuration,
     Self.MinSpeechDuration, Self.WindowSize, Self.MaxSpeechDuration
    ]);
end;

class operator TSherpaOnnxSileroVadModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxSileroVadModelConfig);
begin
  Dest.Threshold := 0.5;
  Dest.MinSilenceDuration := 0.5;
  Dest.MinSpeechDuration := 0.25;
  Dest.WindowSize := 512;
  Dest.MaxSpeechDuration := 5.0;
end;

function TSherpaOnnxVadModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxVadModelConfig(' +
    'SileroVad := %s, ' +
    'SampleRate := %d, ' +
    'NumThreads := %d, ' +
    'Provider := %s, ' +
    'Debug := %s' +
    ')',
    [Self.SileroVad.ToString, Self.SampleRate, Self.NumThreads, Self.Provider,
     Self.Debug.ToString
    ]);
end;

class operator TSherpaOnnxVadModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxVadModelConfig);
begin
  Dest.SampleRate := 16000;
  Dest.NumThreads := 1;
  Dest.Provider := 'cpu';
  Dest.Debug := False;
end;

class operator TSherpaOnnxFeatureConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxFeatureConfig);
begin
  Dest.SampleRate := 16000;
  Dest.FeatureDim := 80;
end;

class operator TSherpaOnnxOnlineCtcFstDecoderConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOnlineCtcFstDecoderConfig);
begin
  Dest.MaxActive := 3000;
end;

class operator TSherpaOnnxOnlineRecognizerConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOnlineRecognizerConfig);
begin
  Dest.DecodingMethod := 'greedy_search';
  Dest.EnableEndpoint := False;
  Dest.Rule1MinTrailingSilence := 2.4;
  Dest.Rule2MinTrailingSilence := 1.2;
  Dest.Rule3MinUtteranceLength := 20;
  Dest.HotwordsScore := 1.5;
  Dest.BlankPenalty := 0;
end;

class operator TSherpaOnnxOnlineModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOnlineModelConfig);
begin
  Dest.NumThreads := 1;
  Dest.Provider := 'cpu';
  Dest.Debug := False;
end;

class operator TSherpaOnnxOfflineWhisperModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineWhisperModelConfig);
begin
  Dest.Task := 'transcribe';
  Dest.TailPaddings := -1;
end;

class operator TSherpaOnnxOfflineLMConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineLMConfig);
begin
  Dest.Scale := 1.0;
end;

class operator TSherpaOnnxOfflineSenseVoiceModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineSenseVoiceModelConfig);
begin
  Dest.UseItn := True;
end;

class operator TSherpaOnnxOfflineModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineModelConfig);
begin
  Dest.NumThreads := 1;
  Dest.Debug := False;
  Dest.Provider := 'cpu';
end;

class operator TSherpaOnnxOfflineRecognizerConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineRecognizerConfig);
begin
  Dest.DecodingMethod := 'greedy_search';
  Dest.MaxActivePaths := 4;
  Dest.HotwordsScore := 1.5;
  Dest.BlankPenalty := 0;
end;

constructor TSherpaOnnxCircularBuffer.Create(Capacity: Integer);
begin
  Self.Handle := SherpaOnnxCreateCircularBuffer(Capacity);
end;

destructor TSherpaOnnxCircularBuffer.Destroy;
begin
  SherpaOnnxDestroyCircularBuffer(Self.Handle);
  Self.Handle := nil;
end;

procedure TSherpaOnnxCircularBuffer.Push(Samples: array of Single);
begin
  SherpaOnnxCircularBufferPush(Self.Handle, pcfloat(Samples), Length(Samples));
end;

procedure TSherpaOnnxCircularBuffer.Push(Samples: pcfloat; N: Integer);
begin
  SherpaOnnxCircularBufferPush(Self.Handle, Samples, N);
end;

function TSherpaOnnxCircularBuffer.Get(StartIndex: Integer; N: Integer): TSherpaOnnxSamplesArray;
var
  P: pcfloat;
  I: Integer;
begin
  P := SherpaOnnxCircularBufferGet(Self.Handle, StartIndex, N);

  Result := nil;

  SetLength(Result, N);

  for I := Low(Result) to High(Result) do
    Result[I] := P[I];

  SherpaOnnxCircularBufferFree(P);
end;

procedure TSherpaOnnxCircularBuffer.Pop(N: Integer);
begin
  SherpaOnnxCircularBufferPop(Self.Handle, N);
end;

procedure TSherpaOnnxCircularBuffer.Reset;
begin
  SherpaOnnxCircularBufferReset(Self.Handle);
end;

function TSherpaOnnxCircularBuffer.Size: Integer;
begin
  Result := SherpaOnnxCircularBufferSize(Self.Handle);
end;

function TSherpaOnnxCircularBuffer.Head: Integer;
begin
  Result := SherpaOnnxCircularBufferHead(Self.Handle);
end;

constructor TSherpaOnnxVoiceActivityDetector.Create(Config: TSherpaOnnxVadModelConfig; BufferSizeInSeconds: Single);
var
  C: SherpaOnnxVadModelConfig ;
begin
  C := Default(SherpaOnnxVadModelConfig);
  Self._Config := Config;

  C.SileroVad.Model := PAnsiChar(Config.SileroVad.Model);
  C.SileroVad.Threshold := Config.SileroVad.Threshold;
  C.SileroVad.MinSilenceDuration := Config.SileroVad.MinSilenceDuration;
  C.SileroVad.MinSpeechDuration := Config.SileroVad.MinSpeechDuration;
  C.SileroVad.WindowSize := Config.SileroVad.WindowSize;
  C.SileroVad.MaxSpeechDuration := Config.SileroVad.MaxSpeechDuration;

  C.SampleRate := Config.SampleRate;
  C.NumThreads := Config.NumThreads;
  C.Provider := PAnsiChar(Config.Provider);
  C.Debug := Ord(Config.Debug);

  Self.Handle := SherpaOnnxCreateVoiceActivityDetector(@C, BufferSizeInSeconds);
end;

destructor TSherpaOnnxVoiceActivityDetector.Destroy;
begin
  SherpaOnnxDestroyVoiceActivityDetector(Self.Handle);
  Self.Handle := nil;
end;

procedure TSherpaOnnxVoiceActivityDetector.AcceptWaveform(Samples: array of Single);
begin
  SherpaOnnxVoiceActivityDetectorAcceptWaveform(Self.Handle, pcfloat(Samples), Length(Samples));
end;

procedure TSherpaOnnxVoiceActivityDetector.AcceptWaveform(Samples: array of Single; Offset: Integer; N: Integer);
begin
  if Offset + N > Length(Samples) then
    begin
      WriteLn(Format('Invalid arguments!. Array length: %d, Offset: %d, N: %d',
        [Length(Samples), Offset, N]
      ));
      Exit;
    end;

  SherpaOnnxVoiceActivityDetectorAcceptWaveform(Self.Handle,
    pcfloat(Samples) + Offset, N);
end;

function TSherpaOnnxVoiceActivityDetector.IsEmpty: Boolean;
begin
  Result := SherpaOnnxVoiceActivityDetectorEmpty(Self.Handle) = 1;
end;

function TSherpaOnnxVoiceActivityDetector.IsDetected: Boolean;
begin
  Result := SherpaOnnxVoiceActivityDetectorDetected(Self.Handle) = 1;
end;

procedure TSherpaOnnxVoiceActivityDetector.Pop;
begin
  SherpaOnnxVoiceActivityDetectorPop(Self.Handle);
end;

procedure TSherpaOnnxVoiceActivityDetector.Clear;
begin
  SherpaOnnxVoiceActivityDetectorClear(Self.Handle);
end;

function TSherpaOnnxVoiceActivityDetector.Front: TSherpaOnnxSpeechSegment;
var
  P: PSherpaOnnxSpeechSegment;
  I: Integer;
begin
  P := SherpaOnnxVoiceActivityDetectorFront(Self.Handle);
  Result.Start := P^.Start;
  Result.Samples := nil;
  SetLength(Result.Samples, P^.N);

  for I := Low(Result.Samples) to High(Result.Samples) do
    Result.Samples[I] := P^.Samples[I];

  SherpaOnnxDestroySpeechSegment(P);
end;

procedure TSherpaOnnxVoiceActivityDetector.Reset;
begin
  SherpaOnnxVoiceActivityDetectorReset(Self.Handle);
end;

procedure TSherpaOnnxVoiceActivityDetector.Flush;
begin
  SherpaOnnxVoiceActivityDetectorFlush(Self.Handle);
end;

function TSherpaOnnxOfflineTtsVitsModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineTtsVitsModelConfig(' +
    'Model := %s, ' +
    'Lexicon := %s, ' +
    'Tokens := %s, ' +
    'DataDir := %s, ' +
    'NoiseScale := %.2f, ' +
    'NoiseScaleW := %.2f, ' +
    'LengthScale := %.2f, ' +
    'DictDir := %s' +
    ')',
    [Self.Model, Self.Lexicon, Self.Tokens, Self.DataDir, Self.NoiseScale,
     Self.NoiseScaleW, Self.LengthScale, Self.DictDir
    ]);
end;

class operator TSherpaOnnxOfflineTtsVitsModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineTtsVitsModelConfig);
begin
  Dest.NoiseScale := 0.667;
  Dest.NoiseScaleW := 0.8;
  Dest.LengthScale := 1.0;
end;

function TSherpaOnnxOfflineTtsModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineTtsModelConfig(' +
    'Vits := %s, ' +
    'NumThreads := %d, ' +
    'Debug := %s, ' +
    'Provider := %s' +
    ')',
    [Self.Vits.ToString, Self.NumThreads, Self.Debug.ToString, Self.Provider
    ]);
end;

class operator TSherpaOnnxOfflineTtsModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineTtsModelConfig);
begin
  Dest.NumThreads := 1;
  Dest.Debug := False;
  Dest.Provider := 'cpu';
end;

function TSherpaOnnxOfflineTtsConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineTtsConfig(' +
    'Model := %s, ' +
    'RuleFsts := %s, ' +
    'MaxNumSentences := %d, ' +
    'RuleFars := %s' +
    ')',
    [Self.Model.ToString, Self.RuleFsts, Self.MaxNumSentences, Self.RuleFars
    ]);
end;

class operator TSherpaOnnxOfflineTtsConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineTtsConfig);
begin
  Dest.MaxNumSentences := 1;
end;

constructor TSherpaOnnxOfflineTts.Create(Config: TSherpaOnnxOfflineTtsConfig);
var
  C: SherpaOnnxOfflineTtsConfig;
begin
  C := Default(SherpaOnnxOfflineTtsConfig);
  Self._Config := Config;

  C.Model.Vits.Model := PAnsiChar(Config.Model.Vits.Model);
  C.Model.Vits.Lexicon := PAnsiChar(Config.Model.Vits.Lexicon);
  C.Model.Vits.Tokens := PAnsiChar(Config.Model.Vits.Tokens);
  C.Model.Vits.DataDir := PAnsiChar(Config.Model.Vits.DataDir);
  C.Model.Vits.NoiseScale := Config.Model.Vits.NoiseScale;
  C.Model.Vits.NoiseScaleW := Config.Model.Vits.NoiseScaleW;
  C.Model.Vits.LengthScale := Config.Model.Vits.LengthScale;
  C.Model.Vits.DictDir := PAnsiChar(Config.Model.Vits.DictDir);

  C.Model.NumThreads := Config.Model.NumThreads;
  C.Model.Provider := PAnsiChar(Config.Model.Provider);
  C.Model.Debug := Ord(Config.Model.Debug);

  C.RuleFsts := PAnsiChar(Config.RuleFsts);
  C.MaxNumSentences := Config.MaxNumSentences;
  C.RuleFars := PAnsiChar(Config.RuleFars);

  Self.Handle := SherpaOnnxCreateOfflineTts(@C);

  Self.SampleRate := SherpaOnnxOfflineTtsSampleRate(Self.Handle);
  Self.NumSpeakers := SherpaOnnxOfflineTtsNumSpeakers(Self.Handle);
end;

destructor TSherpaOnnxOfflineTts.Destroy;
begin
  SherpaOnnxDestroyOfflineTts(Self.Handle);
  Self.Handle := nil;
end;

function TSherpaOnnxOfflineTts.Generate(Text: AnsiString; SpeakerId: Integer;
  Speed: Single): TSherpaOnnxGeneratedAudio;
var
  Audio: PSherpaOnnxGeneratedAudio;
  I: Integer;
begin
  Result := Default(TSherpaOnnxGeneratedAudio);

  Audio := SherpaOnnxOfflineTtsGenerate(Self.Handle, PAnsiChar(Text), SpeakerId, Speed);

  SetLength(Result.Samples, Audio^.N);
  Result.SampleRate := Audio^.SampleRate;

  for I := Low(Result.Samples) to High(Result.Samples) do
  begin
    Result.Samples[I] := Audio^.Samples[I];
  end;

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
end;

function TSherpaOnnxOfflineTts.Generate(Text: AnsiString; SpeakerId: Integer;
  Speed: Single;
  Callback: PSherpaOnnxGeneratedAudioCallbackWithArg;
  Arg: Pointer
  ): TSherpaOnnxGeneratedAudio;
var
  Audio: PSherpaOnnxGeneratedAudio;
  I: Integer;
begin
  Result := Default(TSherpaOnnxGeneratedAudio);

  Audio := SherpaOnnxOfflineTtsGenerateWithCallbackWithArg(Self.Handle, PAnsiChar(Text),
    SpeakerId, Speed, Callback, Arg);

  SetLength(Result.Samples, Audio^.N);
  Result.SampleRate := Audio^.SampleRate;

  for I := Low(Result.Samples) to High(Result.Samples) do
  begin
    Result.Samples[I] := Audio^.Samples[I];
  end;

  SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
end;

constructor TSherpaOnnxLinearResampler.Create(SampleRateIn: Integer; SampleRateOut: Integer);
var
  MinFreq: Single;
  LowpassCutoff: Single;
  LowpassFilterWidth: Integer = 6;
begin
  if SampleRateIn > SampleRateOut then
    MinFreq := SampleRateOut
  else
    MinFreq := SampleRateIn;

  LowpassCutoff := 0.99 * 0.5 * MinFreq;

  Self.Handle := SherpaOnnxCreateLinearResampler(SampleRateIn,
    SampleRateOut, LowpassCutoff, LowpassFilterWidth);
  Self.InputSampleRate := SampleRateIn;
  Self.OutputSampleRate := SampleRateOut;
end;

destructor TSherpaOnnxLinearResampler.Destroy;
begin
  SherpaOnnxDestroyLinearResampler(Self.Handle);
  Self.Handle := nil;
end;

function TSherpaOnnxLinearResampler.Resample(Samples: pcfloat;
  N: Integer; Flush: Boolean): TSherpaOnnxSamplesArray;
var
  P: PSherpaOnnxResampleOut;
  I: Integer;
begin
  Result := Default(TSherpaOnnxSamplesArray);
  P := SherpaOnnxLinearResamplerResample(Self.Handle, Samples, N, Ord(Flush));
  SetLength(Result, P^.N);

  for I := Low(Result) to High(Result) do
    Result[I] := P^.Samples[I];

  SherpaOnnxLinearResamplerResampleFree(P);
end;

function TSherpaOnnxLinearResampler.Resample(Samples: array of Single; Flush: Boolean): TSherpaOnnxSamplesArray;
begin
  Result := Self.Resample(pcfloat(Samples), Length(Samples), Flush);
end;

procedure TSherpaOnnxLinearResampler.Reset;
begin
  SherpaOnnxLinearResamplerReset(Self.Handle);
end;

function TSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(' +
    'Model := %s)',[Self.Model]);
end;

function TSherpaOnnxOfflineSpeakerSegmentationModelConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig(' +
    'Pyannote := %s, ' +
    'NumThreads := %d, ' +
    'Debug := %s, ' +
    'Provider := %s)',
    [Self.Pyannote.ToString, Self.NumThreads,
     Self.Debug.ToString, Self.Provider]);
end;

class operator TSherpaOnnxOfflineSpeakerSegmentationModelConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineSpeakerSegmentationModelConfig);
begin
  Dest.NumThreads := 1;
  Dest.Debug := False;
  Dest.Provider := 'cpu';
end;

function TSherpaOnnxFastClusteringConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxFastClusteringConfig(' +
    'NumClusters := %d, Threshold := %.3f)',
    [Self.NumClusters, Self.Threshold]);
end;

class operator TSherpaOnnxFastClusteringConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxFastClusteringConfig);
begin
  Dest.NumClusters := -1;
  Dest.Threshold := 0.5;
end;

function TSherpaOnnxSpeakerEmbeddingExtractorConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxSpeakerEmbeddingExtractorConfig(' +
    'Model := %s, '+
    'NumThreads := %d, '+
    'Debug := %s, '+
    'Provider := %s)',
    [Self.Model, Self.NumThreads, Self.Debug.ToString, Self.Provider]);
end;

class operator TSherpaOnnxSpeakerEmbeddingExtractorConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxSpeakerEmbeddingExtractorConfig);
begin
  Dest.NumThreads := 1;
  Dest.Debug := False;
  Dest.Provider := 'cpu';
end;

function TSherpaOnnxOfflineSpeakerDiarizationConfig.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineSpeakerDiarizationConfig(' +
    'Segmentation := %s, '+
    'Embedding := %s, '+
    'Clustering := %s, '+
    'MinDurationOn := %.3f, '+
    'MinDurationOff := %.3f)',
    [Self.Segmentation.ToString, Self.Embedding.ToString,
     Self.Clustering.ToString, Self.MinDurationOn, Self.MinDurationOff]);
end;

class operator TSherpaOnnxOfflineSpeakerDiarizationConfig.Initialize({$IFDEF FPC}var{$ELSE}out{$ENDIF} Dest: TSherpaOnnxOfflineSpeakerDiarizationConfig);
begin
  Dest.MinDurationOn := 0.2;
  Dest.MinDurationOff := 0.5;
end;

function TSherpaOnnxOfflineSpeakerDiarizationSegment.ToString: AnsiString;
begin
  Result := Format('TSherpaOnnxOfflineSpeakerDiarizationSegment(' +
    'Start := %.3f, '+
    'Stop := %.3f, '+
    'Speaker := %d)',
    [Self.Start, Self.Stop, Self.Speaker]);
end;

constructor TSherpaOnnxOfflineSpeakerDiarization.Create(Config: TSherpaOnnxOfflineSpeakerDiarizationConfig);
var
  C: SherpaOnnxOfflineSpeakerDiarizationConfig;
begin
  C := Default(SherpaOnnxOfflineSpeakerDiarizationConfig);
  C.Segmentation.Pyannote.Model := PAnsiChar(Config.Segmentation.Pyannote.Model);
  C.Segmentation.NumThreads := Config.Segmentation.NumThreads;
  C.Segmentation.Debug := Ord(Config.Segmentation.Debug);
  C.Segmentation.Provider := PAnsiChar(Config.Segmentation.Provider);

  C.Embedding.Model := PAnsiChar(Config.Embedding.Model);
  C.Embedding.NumThreads := Config.Embedding.NumThreads;
  C.Embedding.Debug := Ord(Config.Embedding.Debug);
  C.Embedding.Provider := PAnsiChar(Config.Embedding.Provider);

  C.Clustering.NumClusters := Config.Clustering.NumClusters;
  C.Clustering.Threshold := Config.Clustering.Threshold;

  C.MinDurationOn := Config.MinDurationOn;
  C.MinDurationOff := Config.MinDurationOff;

  Self.Handle := SherpaOnnxCreateOfflineSpeakerDiarization(@C);
  Self._Config := Config;
  Self.SampleRate :=  0;

  if Self.Handle <> nil then
    begin
      Self.SampleRate := SherpaOnnxOfflineSpeakerDiarizationGetSampleRate(Self.Handle);
    end;
end;

destructor TSherpaOnnxOfflineSpeakerDiarization.Destroy;
begin
  SherpaOnnxDestroyOfflineSpeakerDiarization(Self.Handle);
  Self.Handle := nil;
end;

procedure TSherpaOnnxOfflineSpeakerDiarization.SetConfig(Config: TSherpaOnnxOfflineSpeakerDiarizationConfig);
var
  C: SherpaOnnxOfflineSpeakerDiarizationConfig;
begin
  C := Default(SherpaOnnxOfflineSpeakerDiarizationConfig);

  C.Clustering.NumClusters := Config.Clustering.NumClusters;
  C.Clustering.Threshold := Config.Clustering.Threshold;

  SherpaOnnxOfflineSpeakerDiarizationSetConfig(Self.Handle, @C);
end;

function TSherpaOnnxOfflineSpeakerDiarization.Process(Samples: array of Single): TSherpaOnnxOfflineSpeakerDiarizationSegmentArray;
var
  R: Pointer;
  NumSegments: Integer;
  I: Integer;
  Segments: PSherpaOnnxOfflineSpeakerDiarizationSegment;
begin
  Result := nil;

  R := SherpaOnnxOfflineSpeakerDiarizationProcess(Self.Handle, pcfloat(Samples), Length(Samples));
  if R = nil then
    begin
      Exit
    end;
  NumSegments := SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(R);

  Segments := SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(R);

  SetLength(Result, NumSegments);
  for I := Low(Result) to High(Result) do
    begin
      Result[I].Start := Segments[I].Start;
      Result[I].Stop := Segments[I].Stop;
      Result[I].Speaker := Segments[I].Speaker;
    end;

  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(Segments);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(R);
end;

function TSherpaOnnxOfflineSpeakerDiarization.Process(Samples: array of Single;
  callback: PSherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArg): TSherpaOnnxOfflineSpeakerDiarizationSegmentArray;
var
  R: Pointer;
  NumSegments: Integer;
  I: Integer;
  Segments: PSherpaOnnxOfflineSpeakerDiarizationSegment;
begin
  Result := nil;

  R := SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg(Self.Handle, pcfloat(Samples), Length(Samples), callback);
  if R = nil then
    begin
      Exit
    end;
  NumSegments := SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments(R);

  Segments := SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime(R);

  SetLength(Result, NumSegments);
  for I := Low(Result) to High(Result) do
    begin
      Result[I].Start := Segments[I].Start;
      Result[I].Stop := Segments[I].Stop;
      Result[I].Speaker := Segments[I].Speaker;
    end;

  SherpaOnnxOfflineSpeakerDiarizationDestroySegment(Segments);
  SherpaOnnxOfflineSpeakerDiarizationDestroyResult(R);
end;

end.
