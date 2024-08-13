unit Unit1;

{$mode objfpc}{$H+}

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
  my_worker;

function CreateVad(): TSherpaOnnxVoiceActivityDetector;
var
  Config: TSherpaOnnxVadModelConfig;

  SampleRate: Integer;
  WindowSize: Integer;
begin
  Initialize(Config);

  SampleRate := 16000; {Please don't change it unless you know the details}
  WindowSize := 512; {Please don't change it unless you know the details}

  Config.SileroVad.Model := './silero_vad.onnx';
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

function CreateOfflineRecognizerWhisper(): TSherpaOnnxOfflineRecognizer;
var
  Config: TSherpaOnnxOfflineRecognizerConfig;
begin
  Initialize(Config);

  Config.ModelConfig.Whisper.Encoder := './whisper-encoder.onnx';
  Config.ModelConfig.Whisper.Decoder := './whisper-decoder.onnx';
  Config.ModelConfig.Tokens := './tokens.txt';
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
begin
  if not FileExists('./silero_vad.onnx') then
    begin
      ShowMessage('./silero_vad.onnx does not exist! Please download it from' +
        sLineBreak + 'https://github.com/k2-fsa/sherpa-onnx/tree/asr-models'
      );
      Exit;
    end;
  Self.Vad := CreateVad();

  if not FileExists('./tokens.txt') then
    begin
      ShowMessage('./tokens.txt not found. Please download a non-streaming ASR model first!');
      Exit;
    end;

  if FileExists('./whisper-encoder.onnx') and FileExists('./whisper-decoder.onnx') then
    OfflineRecognizer := CreateOfflineRecognizerWhisper()
  else
    begin
      ShowMessage('Please download at least one non-streaming speech recognition model first.');
      Exit;
    end;

   MessageDlg('Congrat! Model initialized succesfully!', mtInformation, [mbOk], 0);
   FileNameEdt.Enabled := True;
   SelectFileBtn.Enabled := True;
   InitBtn.Enabled := False;
end;

end.

