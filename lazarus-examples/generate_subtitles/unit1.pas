unit Unit1;

{$mode objfpc}{$H+}

{$IFDEF DARWIN}
{$modeswitch objectivec1}   {For getting resource directory}
{$ENDIF}

interface

uses
  Classes, SysUtils, StrUtils, Forms, Controls,
  Graphics, Dialogs, StdCtrls,
  sherpa_onnx, ComCtrls;

type

  { TForm1 }

  TForm1 = class(TForm)
    InitBtn: TButton;
    ProgressBar: TProgressBar;
    ResultMemo: TMemo;
    StartBtn: TButton;
    SelectFileDlg: TOpenDialog;
    SelectFileBtn: TButton;
    FileNameEdt: TEdit;
    ProgressLabel: TLabel;
    procedure FileNameEdtChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure InitBtnClick(Sender: TObject);
    procedure SelectFileBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure StartBtnClick(Sender: TObject);
  private

  public
    procedure UpdateResult(
      Msg: AnsiString;
      StartTime: Single;
      StopTime: Single;
      TotalDuration: Single);
    procedure UpdateProgress(StopTime: Single; TotalDuration: Single);
  public
    Vad: TSherpaOnnxVoiceActivityDetector;
    OfflineRecognizer: TSherpaOnnxOfflineRecognizer;
  end;

var
  Form1: TForm1;

implementation

uses
  my_worker
  {$IFDEF DARWIN}
  ,MacOSAll
  ,CocoaAll
  {$ENDIF}
  ;
{See https://wiki.lazarus.freepascal.org/Locating_the_macOS_application_resources_directory}

{$IFDEF DARWIN}
{Note: The returned path contains a trailing /}
function GetResourcesPath(): AnsiString;
var
  pathStr: shortstring;
  status: Boolean = false;
begin
  status := CFStringGetPascalString(CFStringRef(NSBundle.mainBundle.resourcePath), @pathStr, 255, CFStringGetSystemEncoding());

  if status = true then
    Result := pathStr + PathDelim
  else
    raise Exception.Create('Error in GetResourcesPath()');
end;
{$ENDIF}

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
  Config.SileroVad.MinSpeechDuration := 0.5;
  Config.SileroVad.MinSilenceDuration := 0.5;
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

{$R *.lfm}

{ TForm1 }

procedure TForm1.FormCreate(Sender: TObject);
begin
  StartBtn.Enabled := False;
  SelectFileDlg.Filter := 'All Files|*.wav';
  FileNameEdt.Enabled := False;
  SelectFileBtn.Enabled := False;
  ResultMemo.Lines.Add('1. It supports only 1 channel, 16-bit, 16000Hz wav files');
  ResultMemo.Lines.Add('2. There should be no Chinese characters in the file path.');

  ProgressBar.Position := 0;
  ProgressLabel.Caption := '';
end;

procedure TForm1.StartBtnClick(Sender: TObject);
begin
  if StartBtn.Caption = 'Stop' then
    begin
      if (MyWorkerThread <> nil) and not MyWorkerThread.Finished then
        MyWorkerThread.Terminate;

      StartBtn.Caption := 'Start';
      Exit;
    end;

  ResultMemo.Lines.Clear();
  ResultMemo.Lines.Add('Start processing');

  ProgressBar.Position := 0;
  ProgressLabel.Caption := Format('%d%%', [ProgressBar.Position]);

  MyWorkerThread := TMyWorkerThread.Create(False, FileNameEdt.Text);

  StartBtn.Caption := 'Stop';
end;

procedure TForm1.SelectFileBtnClick(Sender: TObject);
begin
  if SelectFileDlg.Execute then
    begin
      FileNameEdt.Text := SelectFileDlg.FileName;
    end;
end;

procedure TForm1.FileNameEdtChange(Sender: TObject);
begin
  if FileExists(FileNameEdt.Text) then
    StartBtn.Enabled := True
  else
    StartBtn.Enabled := False;
end;

procedure TForm1.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  if (MyWorkerThread <> nil) and not MyWorkerThread.Finished then
    begin
      MyWorkerThread.Terminate;
      MyWorkerThread.WaitFor;
    end;
  FreeAndNil(Vad);
  FreeAndNil(OfflineRecognizer);
end;

procedure TForm1.UpdateProgress(StopTime: Single; TotalDuration: Single);
var
  Percent: Single;
begin
  if (StopTime <> 0) and (TotalDuration <> 0) then
    begin
      Percent := StopTime / TotalDuration * 100;
      ProgressBar.Position := Round(Percent);
      ProgressLabel.Caption := Format('%d%%', [ProgressBar.Position]);
    end;
end;

procedure TForm1.UpdateResult(
  Msg: AnsiString;
  StartTime: Single;
  StopTime: Single;
  TotalDuration: Single);
var
  NewResult: AnsiString;
begin
  UpdateProgress(StopTime, TotalDuration);

  if (Msg = 'DONE!') or
     (Msg = 'Cancelled!') or
     EndsStr('16-bit encoded wave files', Msg) or
     EndsStr('. Please select a new file', Msg) then
    begin
      Form1.StartBtn.Caption := 'Start';
      NewResult := Msg;
    end
  else
    begin
      NewResult := Format('%.3f -- %.3f  %s', [StartTime, StopTime, Msg]);
    end;

  if Msg = 'DONE!' then
    begin
      ProgressBar.Position := 100;

      ProgressLabel.Caption := '100%';
    end;

  Form1.ResultMemo.Lines.Add(NewResult);
end;

procedure TForm1.InitBtnClick(Sender: TObject);
var
  Msg: AnsiString;
  ModelDir: AnsiString;
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
  {$IFDEF DARWIN}
    ModelDir := GetResourcesPath;
  {$ELSE}
    ModelDir := './';
  {$ENDIF}

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
      ShowMessage(VadFilename + ' does not exist! Please download it from' +
        sLineBreak + 'https://github.com/k2-fsa/sherpa-onnx/tree/asr-models'
      );
      Exit;
    end;

  Self.Vad := CreateVad(VadFilename);

  if not FileExists(Tokens) then
    begin
      ShowMessage(Tokens + ' not found. Please download a non-streaming ASR model first!');
      Exit;
    end;

  if FileExists(WhisperEncoder) and FileExists(WhisperDecoder) then
    begin
      OfflineRecognizer := CreateOfflineRecognizerWhisper(Tokens, WhisperEncoder, WhisperDecoder);
      Msg := 'Whisper';
    end
  else if FileExists(SenseVoice) then
    begin
      OfflineRecognizer := CreateOfflineRecognizerSenseVoice(Tokens, SenseVoice);
      Msg := 'SenseVoice';
    end
  else if FileExists(Paraformer) then
    begin
      OfflineRecognizer := CreateOfflineRecognizerParaformer(Tokens, Paraformer);
      Msg := 'Paraformer';
    end
  else if FileExists(TeleSpeech) then
    begin
      OfflineRecognizer := CreateOfflineRecognizerTeleSpeech(Tokens, TeleSpeech);
      Msg := 'TeleSpeech';
    end
  else if FileExists(TransducerEncoder) and FileExists(TransducerDecoder) and FileExists(TransducerJoiner) then
      begin
        OfflineRecognizer := CreateOfflineRecognizerTransducer(Tokens,
          TransducerEncoder, TransducerDecoder, TransducerJoiner, 'transducer');
        Msg := 'Zipformer transducer';
      end
  else if FileExists(NeMoTransducerEncoder) and FileExists(NeMoTransducerDecoder) and FileExists(NeMoTransducerJoiner) then
      begin
        OfflineRecognizer := CreateOfflineRecognizerTransducer(Tokens,
          NeMoTransducerEncoder, NeMoTransducerDecoder, NeMoTransducerJoiner, 'nemo_transducer');
        Msg := 'NeMo transducer';
      end
  else
    begin
      ShowMessage('Please download at least one non-streaming speech recognition model first.');
      Exit;
    end;

   MessageDlg('Congrat! The ' + Msg + ' model is initialized succesfully!', mtInformation, [mbOk], 0);
   FileNameEdt.Enabled := True;
   SelectFileBtn.Enabled := True;
   InitBtn.Enabled := False;
end;

end.

