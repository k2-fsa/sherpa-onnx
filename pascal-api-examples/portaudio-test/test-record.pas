{ Copyright (c)  2024  Xiaomi Corporation }
{
This file shows how to use portaudio for recording.

It records for 10 seconds and saves the audio samples to ./record.wav
}
program main;

{$mode objfpc}

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
  SampleRate: Double;
  Stream: PPaStream;

  Buffer: TSherpaOnnxCircularBuffer;
  AllSamples: TSherpaOnnxSamplesArray;

function RecordCallback(
      input: Pointer; output: Pointer;
      frameCount: culong;
      timeInfo: PPaStreamCallbackTimeInfo;
      statusFlags: TPaStreamCallbackFlags;
      userData: Pointer ): cint; cdecl;
begin
  Buffer.Push(pcfloat(input), frameCount);
  Result := paContinue;
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

  DeviceIndex := Pa_GetDefaultInputDevice;

  if DeviceIndex = paNoDevice then
    begin
      WriteLn('No default input device found');
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
  WriteLn(' Max input channels ', Pa_GetDeviceInfo(DeviceIndex)^.MaxInputChannels);

  Initialize(Param);
  Param.Device := DeviceIndex;
  Param.ChannelCount := 1;
  Param.SampleFormat := paFloat32;
  param.SuggestedLatency := Pa_GetDeviceInfo(DeviceIndex)^.DefaultHighInputLatency;
  param.HostApiSpecificStreamInfo := nil;

  SampleRate := 48000;
  Buffer := TSherpaOnnxCircularBuffer.Create(Round(SampleRate) * 20);

  Status := Pa_OpenStream(stream, @Param, nil, SampleRate, paFramesPerBufferUnspecified, paNoFlag,
    PPaStreamCallback(@RecordCallback), nil);

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

  WriteLn('Please speak! It will exit after 10 seconds.');
  Pa_Sleep(10000);  {sleep for 10 seconds }

  Status := Pa_CloseStream(stream);
  if Status <> paNoError then
    begin
      WriteLn('Failed to close stream, ', Pa_GetErrorText(Status));
      Exit;
    end;

  AllSamples := Buffer.Get(0, Buffer.Size);

  SherpaOnnxWriteWave('record.wav', AllSamples, Round(SampleRate));
  WriteLn('Saved to record.wav');

  Status := Pa_Terminate;
  if Status <> paNoError then
    begin
      WriteLn('Failed to deinitialize portaudio, ', Pa_GetErrorText(Status));
      Exit;
    end;
end.

