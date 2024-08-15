unit Unit1;

{$mode objfpc}{$H+}

{$IFDEF DARWIN}
{$modeswitch objectivec1}   {For getting resource directory}
{$ENDIF}

interface

uses
  Classes, SysUtils, Forms, Controls,
  Graphics, Dialogs, StdCtrls,
  sherpa_onnx;

type

  { TForm1 }

  TForm1 = class(TForm)
    InitBtn: TButton;
    ResultMemo: TMemo;
    StartBtn: TButton;
    SelectFileDlg: TOpenDialog;
    SelectFileBtn: TButton;
    FileNameEdt: TEdit;
    procedure FileNameEdtChange(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure InitBtnClick(Sender: TObject);
    procedure SelectFileBtnClick(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure StartBtnClick(Sender: TObject);
  private

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
  if (MyWorkerThread <> nil) and not MyWorkerThread.Finished then
    begin
     ResultMemo.Lines.Add('Stop it');
     MyWorkerThread.Terminate;
    end;


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

procedure TForm1.InitBtnClick(Sender: TObject);
var
  Msg: AnsiString;
  ModelDir: AnsiString;
  VadFilename: AnsiString;
  Tokens: AnsiString;

  WhisperEncoder: AnsiString;
  WhisperDecoder: AnsiString;

  SenseVoice: AnsiString;
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

