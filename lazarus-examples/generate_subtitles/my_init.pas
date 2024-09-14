unit my_init;

{$mode ObjFPC}{$H+}

interface

uses
  {$IFDEF UNIX}
  cthreads,
  cmem,
  {$ENDIF}
  {$IFDEF HASAMIGA}
  athreads,
  {$ENDIF}
  Classes, SysUtils;

type
  TMyInitThread = class(TThread)
  private
    Status: AnsiString;
    ModelDir: AnsiString;
    procedure ShowStatus;

  protected
    procedure Execute; override;
  public
    Constructor Create(CreateSuspended: Boolean; ModelDirectory: AnsiString);
  end;

var
  MyInitThread: TMyInitThread;

implementation

uses
  unit1, sherpa_onnx;

function CreateVad(VadFilename: AnsiString): TSherpaOnnxVoiceActivityDetector;
var
  Config: TSherpaOnnxVadModelConfig;

  SampleRate: Integer;
  WindowSize: Integer;
begin
  Initialize(Config);

  SampleRate := 16000; {Please don't change it unless you know the details}
  WindowSize := 512; {Please don't change it unless you know the details}

  Config.SileroVad.Model := VadFilename;
  Config.SileroVad.MinSpeechDuration := 0.25;
  Config.SileroVad.MinSilenceDuration := 0.5;
  Config.SileroVad.MaxSpeechDuration := 5.0;
  Config.SileroVad.Threshold := 0.5;
  Config.SileroVad.WindowSize := WindowSize;
  Config.NumThreads:= 2;
  Config.Debug:= True;
  Config.Provider:= 'cpu';
  Config.SampleRate := SampleRate;

  Result := TSherpaOnnxVoiceActivityDetector.Create(Config, 30);
end;

function CreateOfflineRecognizerTransducer(
  Tokens: AnsiString;
  Encoder: AnsiString;
  Decoder: AnsiString;
  Joiner: AnsiString;
  ModelType: AnsiString): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(Config);

  Config.ModelConfig.Transducer.Encoder := Encoder;
  Config.ModelConfig.Transducer.Decoder := Decoder;
  Config.ModelConfig.Transducer.Joiner := Joiner;

  Config.ModelConfig.ModelType := ModelType;
  Config.ModelConfig.Tokens := Tokens;
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 2;
  Config.ModelConfig.Debug := False;

  Result := TSherpaOnnxOfflineRecognizer.Create(Config);
end;

function CreateOfflineRecognizerTeleSpeech(
  Tokens: AnsiString;
  TeleSpeech: AnsiString): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(Config);

  Config.ModelConfig.TeleSpeechCtc := TeleSpeech;

  Config.ModelConfig.Tokens := Tokens;
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 2;
  Config.ModelConfig.Debug := False;

  Result := TSherpaOnnxOfflineRecognizer.Create(Config);
end;

function CreateOfflineRecognizerParaformer(
  Tokens: AnsiString;
  Paraformer: AnsiString): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(Config);

  Config.ModelConfig.Paraformer.Model := Paraformer;

  Config.ModelConfig.Tokens := Tokens;
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 2;
  Config.ModelConfig.Debug := False;

  Result := TSherpaOnnxOfflineRecognizer.Create(Config);
end;

function CreateOfflineRecognizerSenseVoice(
  Tokens: AnsiString;
  SenseVoice: AnsiString): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(Config);

  Config.ModelConfig.SenseVoice.Model := SenseVoice;
  Config.ModelConfig.SenseVoice.Language := 'auto';
  Config.ModelConfig.SenseVoice.UseItn := True;
  Config.ModelConfig.Tokens := Tokens;
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 2;
  Config.ModelConfig.Debug := False;

  Result := TSherpaOnnxOfflineRecognizer.Create(Config);
end;

function CreateOfflineRecognizerWhisper(
  Tokens: AnsiString;
  WhisperEncoder: AnsiString;
  WhisperDecoder: AnsiString): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(Config);

  Config.ModelConfig.Whisper.Encoder := WhisperEncoder;
  Config.ModelConfig.Whisper.Decoder := WhisperDecoder;
  Config.ModelConfig.Tokens := Tokens;
  Config.ModelConfig.Provider := 'cpu';
  Config.ModelConfig.NumThreads := 2;
  Config.ModelConfig.Debug := False;

  Result := TSherpaOnnxOfflineRecognizer.Create(Config);
end;

constructor TMyInitThread.Create(CreateSuspended : boolean; ModelDirectory: AnsiString);
begin
  inherited Create(CreateSuspended);
  ModelDir := ModelDirectory;
  FreeOnTerminate := True;
end;

procedure TMyInitThread.ShowStatus;
begin
  Form1.UpdateInitStatus(Status);
end;

procedure TMyInitThread.Execute;
var
  Msg: AnsiString;
  VadFilename: AnsiString;
  Tokens: AnsiString;

  WhisperEncoder: AnsiString;
  WhisperDecoder: AnsiString;

  SenseVoice: AnsiString;

  Paraformer: AnsiString;

  TeleSpeech: AnsiString;

  TransducerEncoder: AnsiString; // from icefall
  TransducerDecoder: AnsiString;
  TransducerJoiner: AnsiString;

  NeMoTransducerEncoder: AnsiString;
  NeMoTransducerDecoder: AnsiString;
  NeMoTransducerJoiner: AnsiString;
begin
    VadFilename := ModelDir + 'silero_vad.onnx';
    Tokens := ModelDir + 'tokens.txt';

    {
      Please refer to
      https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/export-onnx.html#available-models
      for a list of whisper models.

      In the code, we use the normalized filename whisper-encoder.onnx, whisper-decoder.onnx, and tokens.txt
      You need to rename the existing model files.

      For instance, if you use sherpa-onnx-whisper-tiny.en, you have to do
        mv tiny.en-tokens.txt tokens.txt

        mv tiny.en-encoder.onnx whisper-encoder.onnx
        mv tiny.en-decoder.onnx whisper-decoder.onnx

        // or use the int8.onnx

        mv tiny.en-encoder.int8.onnx whisper-encoder.onnx
        mv tiny.en-decoder.int8.onnx whisper-decoder.onnx
    }
    WhisperEncoder := ModelDir + 'whisper-encoder.onnx';
    WhisperDecoder := ModelDir + 'whisper-decoder.onnx';


    {
      Please refer to
      https://k2-fsa.github.io/sherpa/onnx/sense-voice/pretrained.html#pre-trained-models
      to download models for SenseVoice.

      In the code, we use the normalized model name sense-voice.onnx. You have
      to rename the downloaded model files.

      For example, you need to use

          mv model.onnx sense-voice.onnx

          // or use the int8.onnx
          mv model.int8.onnx sense-voice.onnx
    }

    SenseVoice := ModelDir + 'sense-voice.onnx';

    {
      Please refer to
      https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/index.html
      to download paraformer models.

      Note that you have to rename model.onnx or model.int8.onnx to paraformer.onnx.
      An example is given below for the rename:

        cp model.onnx paraformer.onnx

        // or use int8.onnx
        cp model.int8.onnx paraformer.onnx
    }
    Paraformer := ModelDir + 'paraformer.onnx';


    {
      please refer to
      https://k2-fsa.github.io/sherpa/onnx/pretrained_models/telespeech/models.html
      to download TeleSpeech models.

      Note that you have to rename model files after downloading. The following
      is an example

         mv model.onnx telespeech.onnx

         // or to use int8.onnx

         mv model.int8.onnx telespeech.onnx
    }

    TeleSpeech := ModelDir + 'telespeech.onnx';


    {
      Please refer to
      https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
      to download an icefall offline transducer model. Note that you need to rename the
      model files to transducer-encoder.onnx, transducer-decoder.onnx, and
      transducer-joiner.onnx
    }
    TransducerEncoder := ModelDir + 'transducer-encoder.onnx';
    TransducerDecoder := ModelDir + 'transducer-decoder.onnx';
    TransducerJoiner := ModelDir + 'transducer-joiner.onnx';

    {
      Please visit
      https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models
      to donwload a NeMo transducer model.
    }
    NeMoTransducerEncoder := ModelDir + 'nemo-transducer-encoder.onnx';
    NeMoTransducerDecoder := ModelDir + 'nemo-transducer-decoder.onnx';
    NeMoTransducerJoiner := ModelDir + 'nemo-transducer-joiner.onnx';

    if not FileExists(VadFilename) then
      begin
        Status := VadFilename + ' does not exist! Please download it from' +
          sLineBreak + 'https://github.com/k2-fsa/sherpa-onnx/tree/asr-models';
        Synchronize(@ShowStatus);
        Exit;
      end;

    if Form1.Vad = nil then
      begin
        Form1.Vad := CreateVad(VadFilename);
      end;

    if not FileExists(Tokens) then
      begin
        Status := Tokens + ' not found. Please download a non-streaming ASR model first!';
        Synchronize(@ShowStatus);
        Exit;
      end;

    if FileExists(WhisperEncoder) and FileExists(WhisperDecoder) then
      begin
        Form1.OfflineRecognizer := CreateOfflineRecognizerWhisper(Tokens, WhisperEncoder, WhisperDecoder);
        Msg := 'Whisper';
      end
    else if FileExists(SenseVoice) then
      begin
        Form1.OfflineRecognizer := CreateOfflineRecognizerSenseVoice(Tokens, SenseVoice);
        Msg := 'SenseVoice';
      end
    else if FileExists(Paraformer) then
      begin
        Form1.OfflineRecognizer := CreateOfflineRecognizerParaformer(Tokens, Paraformer);
        Msg := 'Paraformer';
      end
    else if FileExists(TeleSpeech) then
      begin
        Form1.OfflineRecognizer := CreateOfflineRecognizerTeleSpeech(Tokens, TeleSpeech);
        Msg := 'TeleSpeech';
      end
    else if FileExists(TransducerEncoder) and FileExists(TransducerDecoder) and FileExists(TransducerJoiner) then
        begin
          Form1.OfflineRecognizer := CreateOfflineRecognizerTransducer(Tokens,
            TransducerEncoder, TransducerDecoder, TransducerJoiner, 'transducer');
          Msg := 'Zipformer transducer';
        end
    else if FileExists(NeMoTransducerEncoder) and FileExists(NeMoTransducerDecoder) and FileExists(NeMoTransducerJoiner) then
        begin
          Form1.OfflineRecognizer := CreateOfflineRecognizerTransducer(Tokens,
            NeMoTransducerEncoder, NeMoTransducerDecoder, NeMoTransducerJoiner, 'nemo_transducer');
          Msg := 'NeMo transducer';
        end
    else
      begin
        Status := 'Please download at least one non-streaming speech recognition model first.';
        Synchronize(@ShowStatus);
        Exit;
      end;

    Status := 'Congratulations! The ' + Msg + ' model is initialized succesfully!';
    Synchronize(@ShowStatus);
end;

end.

