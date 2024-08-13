unit my_worker;

{$mode ObjFPC}{$H+}

{
See
https://wiki.lazarus.freepascal.org/Multithreaded_Application_Tutorial

https://www.freepascal.org/docs-html/rtl/classes/tthread.html
}

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
  TMyWorkerThread = class(TThread)
  private
    Status: AnsiString;
    procedure ShowStatus;
  protected
    procedure Execute; override;
  public
    WaveFilename: AnsiString;
    Constructor Create(CreateSuspended : boolean; Filename: AnsiString);
  end;

var
  MyWorkerThread: TMyWorkerThread;

implementation

uses
  unit1, sherpa_onnx;

constructor TMyWorkerThread.Create(CreateSuspended : boolean; Filename: AnsiString);
begin
  inherited Create(CreateSuspended);
  WaveFilename := Filename;
  FreeOnTerminate := True;
end;

procedure TMyWorkerThread.ShowStatus;
begin
  if Status = 'DONE!' then
  begin
    Form1.StartBtn.Caption := 'Start';
  end;
  Form1.ResultMemo.Lines.Add(Status);
end;
procedure TMyWorkerThread.Execute;
var
  Wave: TSherpaOnnxWave;
  WindowSize: Integer;
  Offset: Integer;
  SpeechSegment: TSherpaOnnxSpeechSegment;
  StartTime: Single;
  Duration: Single;
  StopTime: Single;

  Stream: TSherpaOnnxOfflineStream;
  RecognitionResult: TSherpaOnnxOfflineRecognizerResult;
begin
  Wave := SherpaOnnxReadWave(WaveFilename);
  if Length(Wave.Samples) = 0 then
    begin
      Status := Format('Failed to read %s. We only support 1 channel, 16000Hz, 16-bit encoded wave files',
        [Wavefilename]);
      Synchronize(@ShowStatus);
      Exit;
    end;
  if Wave.SampleRate <> 16000 then
    begin
      Status := Format('Expected sample rate 16000. Given %d. Please select a new file', [Wave.SampleRate]);
      Synchronize(@ShowStatus);
      Exit;
    end;

  WindowSize := Form1.Vad.Config.SileroVad.WindowSize;
  Offset := 0;
  Form1.Vad.Reset;

  while not Terminated and (Offset + WindowSize <= Length(Wave.Samples)) do
      begin
        Form1.Vad.AcceptWaveform(Wave.Samples, Offset, WindowSize);
        Offset += WindowSize;

        while not Terminated and not Form1.Vad.IsEmpty do
          begin
            SpeechSegment := Form1.Vad.Front;
            Form1.Vad.Pop;
            Stream := Form1.OfflineRecognizer.CreateStream;

            Stream.AcceptWaveform(SpeechSegment.Samples, Wave.SampleRate);
            Form1.OfflineRecognizer.Decode(Stream);
            RecognitionResult := Form1.OfflineRecognizer.GetResult(Stream);

            StartTime := SpeechSegment.Start / Wave.SampleRate;
            Duration := Length(SpeechSegment.Samples) / Wave.SampleRate;
            StopTime := StartTime + Duration;
            Status := Format('%.3f -- %.3f %s', [StartTime, StopTime, RecognitionResult.Text]);
            Synchronize(@ShowStatus);
            FreeAndNil(Stream);
          end;
      end;

    Form1.Vad.Flush;
    while not Terminated and not Form1.Vad.IsEmpty do
      begin
        SpeechSegment := Form1.Vad.Front;
        Form1.Vad.Pop;
        Stream := Form1.OfflineRecognizer.CreateStream;

        Stream.AcceptWaveform(SpeechSegment.Samples, Wave.SampleRate);
        Form1.OfflineRecognizer.Decode(Stream);
        RecognitionResult := Form1.OfflineRecognizer.GetResult(Stream);

        StartTime := SpeechSegment.Start / Wave.SampleRate;
        Duration := Length(SpeechSegment.Samples) / Wave.SampleRate;
        StopTime := StartTime + Duration;
        Status := Format('%.3f -- %.3f %s', [StartTime, StopTime, RecognitionResult.Text]);

        Synchronize(@ShowStatus);
        FreeAndNil(Stream);
      end;

    Status := 'DONE!';
    Synchronize(@ShowStatus);
end;

end.

