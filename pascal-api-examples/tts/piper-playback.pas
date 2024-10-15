{ Copyright (c)  2024  Xiaomi Corporation }
program piper;
{
This file shows how to use the text to speech API of sherpa-onnx
with Piper models.

It generates speech from text and saves it to a wave file.

Note that it plays the audio back as it is still generating.
}

{$mode objfpc}

uses
  {$ifdef unix}
  cthreads,
  {$endif}
  SysUtils,
  dos,
  ctypes,
  portaudio,
  sherpa_onnx;

var
  CriticalSection: TRTLCriticalSection;

  Tts: TSherpaOnnxOfflineTts;
  Audio: TSherpaOnnxGeneratedAudio;
  Resampler: TSherpaOnnxLinearResampler;

  Text: AnsiString;
  Speed: Single = 1.0;  {Use a larger value to speak faster}
  SpeakerId: Integer = 0;
  Buffer: TSherpaOnnxCircularBuffer;
  FinishedGeneration: Boolean = False;
  FinishedPlaying: Boolean = False;

  Version: String;
  EnvStr: String;
  Status: Integer;
  NumDevices: Integer;
  DeviceIndex: Integer;
  DeviceInfo: PPaDeviceInfo;

  { If you get EDivByZero: Division by zero error, please change the sample rate
    to the one supported by your microphone.
  }
  DeviceSampleRate: Integer = 48000;
  I: Integer;
  Param: TPaStreamParameters;
  Stream: PPaStream;
  Wave: TSherpaOnnxWave;

function GenerateCallback(
      Samples: pcfloat; N: cint32;
      Arg: Pointer): cint; cdecl;
begin
  EnterCriticalSection(CriticalSection);
  try
    if Resampler <> nil then
      Buffer.Push(Resampler.Resample(Samples, N, False))
    else
      Buffer.Push(Samples, N);
  finally
    LeaveCriticalSection(CriticalSection);
  end;

  { 1 means to continue generating; 0 means to stop generating. }
  Result := 1;
end;

function PlayCallback(
      input: Pointer; output: Pointer;
      frameCount: culong;
      timeInfo: PPaStreamCallbackTimeInfo;
      statusFlags: TPaStreamCallbackFlags;
      userData: Pointer ): cint; cdecl;
var
  Samples: TSherpaOnnxSamplesArray;
  I: Integer;
begin
  EnterCriticalSection(CriticalSection);
  try
    if Buffer.Size >= frameCount then
      begin
        Samples := Buffer.Get(Buffer.Head, FrameCount);
        Buffer.Pop(FrameCount);
      end
    else if Buffer.Size > 0 then
      begin
        Samples := Buffer.Get(Buffer.Head, Buffer.Size);
        Buffer.Pop(Buffer.Size);
        SetLength(Samples, frameCount);
      end
    else
      SetLength(Samples, frameCount);

    for I := 0 to frameCount - 1 do
      pcfloat(output)[I] := Samples[I];

    if (Buffer.Size > 0) or (not FinishedGeneration) then
      Result := paContinue
    else
      begin
        Result := paComplete;
        FinishedPlaying := True;
      end;
  finally
    LeaveCriticalSection(CriticalSection);
  end;
end;

function GetOfflineTts: TSherpaOnnxOfflineTts;
var
  Config: TSherpaOnnxOfflineTtsConfig;
begin
  Config.Model.Vits.Model := './vits-piper-en_US-libritts_r-medium/en_US-libritts_r-medium.onnx';
  Config.Model.Vits.Tokens := './vits-piper-en_US-libritts_r-medium/tokens.txt';
  Config.Model.Vits.DataDir := './vits-piper-en_US-libritts_r-medium/espeak-ng-data';
  Config.Model.NumThreads := 1;
  Config.Model.Debug := False;
  Config.MaxNumSentences := 1;

  Result := TSherpaOnnxOfflineTts.Create(Config);
end;

begin
  Tts := GetOfflineTts;
  if Tts.GetSampleRate <> DeviceSampleRate then
    Resampler := TSherpaOnnxLinearResampler.Create(Tts.GetSampleRate, DeviceSampleRate);

  Version := String(Pa_GetVersionText);
  WriteLn('Version is ', Version);
  Status := Pa_Initialize;
  if Status <> paNoError then
    begin
      WriteLn('Failed to initialize portaudio, ', Pa_GetErrorText(Status));
      Exit;
    end;

  NumDevices := Pa_GetDeviceCount;
  WriteLn('Num devices: ', NumDevices);

  DeviceIndex := Pa_GetDefaultOutputDevice;

  if DeviceIndex = paNoDevice then
    begin
      WriteLn('No default output device found');
      Pa_Terminate;
      Exit;
    end;

  EnvStr := GetEnv('SHERPA_ONNX_MIC_DEVICE');
  if EnvStr <> '' then
    begin
      DeviceIndex := StrToIntDef(EnvStr, DeviceIndex);
      WriteLn('Use device index from environment variable SHERPA_ONNX_MIC_DEVICE: ', EnvStr);
    end;

  for I := 0 to (NumDevices - 1) do
    begin
      DeviceInfo := Pa_GetDeviceInfo(I);
      if I = DeviceIndex then
        { WriteLn(Format(' * %d %s', [I, DeviceInfo^.Name])) }
        WriteLn(Format(' * %d %s', [I, AnsiString(DeviceInfo^.Name)]))
      else
        WriteLn(Format('   %d %s', [I, AnsiString(DeviceInfo^.Name)]));
    end;

  WriteLn('Use device ', DeviceIndex);
  WriteLn(' Name ', Pa_GetDeviceInfo(DeviceIndex)^.Name);
  WriteLn(' Max output channels ', Pa_GetDeviceInfo(DeviceIndex)^.MaxOutputChannels);

  Initialize(Param);
  Param.Device := DeviceIndex;
  Param.ChannelCount := 1;
  Param.SampleFormat := paFloat32;
  param.SuggestedLatency := Pa_GetDeviceInfo(DeviceIndex)^.DefaultHighOutputLatency;
  param.HostApiSpecificStreamInfo := nil;

  Buffer := TSherpaOnnxCircularBuffer.Create(30 * DeviceSampleRate);


  { Note(fangjun): PortAudio invokes PlayCallback in a separate thread. }
  Status := Pa_OpenStream(stream, nil, @Param, DeviceSampleRate, paFramesPerBufferUnspecified, paNoFlag,
    PPaStreamCallback(@PlayCallback), nil);

  if Status <> paNoError then
    begin
      WriteLn('Failed to open stream, ', Pa_GetErrorText(Status));
      Pa_Terminate;
      Exit;
    end;

  InitCriticalSection(CriticalSection);

  Status := Pa_StartStream(stream);
  if Status <> paNoError then
    begin
      WriteLn('Failed to start stream, ', Pa_GetErrorText(Status));
      Pa_Terminate;
      Exit;
    end;

  WriteLn('There are ', Tts.GetNumSpeakers, ' speakers');

  Text := 'Today as always, men fall into two groups: slaves and free men. Whoever does not have two-thirds of his day for himself, is a slave, whatever he may be: a statesman, a businessman, an official, or a scholar.';

  Audio :=  Tts.Generate(Text, SpeakerId, Speed,
    PSherpaOnnxGeneratedAudioCallbackWithArg(@GenerateCallback), nil);
  FinishedGeneration := True;
  SherpaOnnxWriteWave('./libritts_r-generated.wav', Audio.Samples, Audio.SampleRate);
  WriteLn('Saved to ./libritts_r-generated.wav');

  while not FinishedPlaying do
    Pa_Sleep(100);  {sleep for 0.1 second }
    {TODO(fangjun): Use an event to indicate the play is finished}

  DoneCriticalSection(CriticalSection);

  FreeAndNil(Tts);
  FreeAndNil(Resampler);

  Status := Pa_CloseStream(stream);
  if Status <> paNoError then
    begin
      WriteLn('Failed to close stream, ', Pa_GetErrorText(Status));
      Exit;
    end;

  Status := Pa_Terminate;
  if Status <> paNoError then
    begin
      WriteLn('Failed to deinitialize portaudio, ', Pa_GetErrorText(Status));
      Exit;
    end;
end.

