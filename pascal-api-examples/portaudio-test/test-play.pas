{ Copyright (c)  2024  Xiaomi Corporation }
{
This file shows how to use portaudio for playing.

}
program main;

{$mode objfpc}{$H+}


uses
  portaudio,
  sherpa_onnx,
  dos,
  ctypes,
  SysUtils;

var
  Version: String;
  EnvStr: String;
  Status: Integer;
  NumDevices: Integer;
  DeviceIndex: Integer;
  DeviceInfo: PPaDeviceInfo;
  I: Integer;
  Param: TPaStreamParameters;
  Stream: PPaStream;
  Wave: TSherpaOnnxWave;

  Buffer: TSherpaOnnxCircularBuffer;

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
  if Buffer.Size >= frameCount then
    begin
      Samples := Buffer.Get(Buffer.Head, FrameCount);
      Buffer.Pop(FrameCount);
    end
  else
    begin
      Samples := Buffer.Get(Buffer.Head, Buffer.Size);
      Buffer.Pop(Buffer.Size);
      SetLength(Samples, frameCount);
    end;
  for I := 0 to frameCount - 1 do
    pcfloat(output)[I] := Samples[I];

  if Buffer.Size > 0 then
    Result := paContinue
  else
    Result := paComplete;
end;



begin
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

  Wave := SherpaOnnxReadWave('./record.wav');
  if Wave.Samples = nil then
    begin
      WriteLn('Failed to read ./record.wav');
      Pa_Terminate;
      Exit;
    end;

  Initialize(Param);
  Param.Device := DeviceIndex;
  Param.ChannelCount := 1;
  Param.SampleFormat := paFloat32;
  param.SuggestedLatency := Pa_GetDeviceInfo(DeviceIndex)^.DefaultHighOutputLatency;
  param.HostApiSpecificStreamInfo := nil;

  Buffer := TSherpaOnnxCircularBuffer.Create(Length(Wave.Samples));
  Buffer.Push(Wave.Samples);

  Status := Pa_OpenStream(stream, nil, @Param, Wave.SampleRate, paFramesPerBufferUnspecified, paNoFlag,
    PPaStreamCallback(@PlayCallback), nil);

  if Status <> paNoError then
    begin
      WriteLn('Failed to open stream, ', Pa_GetErrorText(Status));
      Pa_Terminate;
      Exit;
    end;

  Status := Pa_StartStream(stream);
  if Status <> paNoError then
    begin
      WriteLn('Failed to start stream, ', Pa_GetErrorText(Status));
      Pa_Terminate;
      Exit;
    end;

  while Buffer.Size > 0 do
    Pa_Sleep(100);  {sleep for 0.1 second }

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

