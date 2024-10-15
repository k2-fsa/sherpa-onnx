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
    procedure UpdateInitStatus(Status: AnsiString);
  public
    Vad: TSherpaOnnxVoiceActivityDetector;
    OfflineRecognizer: TSherpaOnnxOfflineRecognizer;
  end;

var
  Form1: TForm1;

implementation

uses
  my_worker,
  my_init
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
  if (MyWorkerThread <> nil) and (not MyWorkerThread.Finished) then
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

procedure TForm1.UpdateInitStatus(Status: AnsiString);
begin
  if EndsStr('model is initialized succesfully!', Status) then
    begin
      Form1.ResultMemo.Lines.Add(Status);
      Form1.ResultMemo.Lines.Add('Please select a 16000Hz wave file to generate subtiles');
      Form1.ResultMemo.Lines.Add('You can download some test wave files from https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models');
      Form1.ResultMemo.Lines.Add('For instance:');
      Form1.ResultMemo.Lines.Add('  Chinese test wave: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/lei-jun-test.wav');
      Form1.ResultMemo.Lines.Add('  English test wave: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/Obama.wav');
      FileNameEdt.Enabled := True;
      SelectFileBtn.Enabled := True;

    end
  else
    begin
      ShowMessage(Status);
      Form1.ResultMemo.Lines.Clear();
      Form1.ResultMemo.Lines.Add('Please refer to');
      Form1.ResultMemo.Lines.Add('https://k2-fsa.github.io/sherpa/onnx/lazarus/generate-subtitles.html#download-models');
      Form1.ResultMemo.Lines.Add('for how to download models');

      InitBtn.Enabled := True;
    end;
end;

procedure TForm1.InitBtnClick(Sender: TObject);
var
  ModelDir: AnsiString;
begin
  {$IFDEF DARWIN}
    ModelDir := GetResourcesPath;
  {$ELSE}
    ModelDir := './';
  {$ENDIF}

  Form1.ResultMemo.Lines.Clear();
  ResultMemo.Lines.Add('Initializing the model. Please wait...');
  MyInitThread := TMyInitThread.Create(False, ModelDir);
  InitBtn.Enabled := False;
end;

end.

