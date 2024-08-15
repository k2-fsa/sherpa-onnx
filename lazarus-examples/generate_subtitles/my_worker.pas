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
    StartTime: Single;
    StopTime: Single;
    TotalDuration: Single;
    procedure ShowStatus;
    procedure ShowProgress;
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
  Form1.UpdateResult(Status, StartTime, StopTime, TotalDuration);
end;

procedure TMyWorkerThread.ShowProgress;
begin
  Form1.UpdateProgress(StopTime, TotalDuration);
end;

procedure TMyWorkerThread.Execute;
var
  Wave: TSherpaOnnxWave;
  WindowSize: Integer;
  Offset: Integer;
  SpeechSegment: TSherpaOnnxSpeechSegment;

  Duration: Single;


  Stream: TSherpaOnnxOfflineStream;
  RecognitionResult: TSherpaOnnxOfflineRecognizerResult;
begin
  Wave := SherpaOnnxReadWave(WaveFilename);
  TotalDuration := 0;
  StartTime := 0;
  StopTime := 0;
  if (Wave.Samples = nil) or (Length(Wave.Samples) = 0) then
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
  TotalDuration := Length(Wave.Samples) / Wave.SampleRate;
  WindowSize := Form1.Vad.Config.SileroVad.WindowSize;

  Offset := 0;
  Form1.Vad.Reset;

  while not Terminated and (Offset + WindowSize <= Length(Wave.Samples)) do
      begin
        Form1.Vad.AcceptWaveform(Wave.Samples, Offset, WindowSize);
        Offset += WindowSize;
        StopTime := Offset / Wave.SampleRate;

        if (Offset mod 20480) = 0 then
          Synchronize(@ShowProgress);

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
            Status := RecognitionResult.Text;

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
        Status := RecognitionResult.Text;

        Synchronize(@ShowStatus);
        FreeAndNil(Stream);
      end;

    if Terminated then
      Status := 'Cancelled!'
    else
      Status := 'DONE!';

    Synchronize(@ShowStatus);
end;

end.

