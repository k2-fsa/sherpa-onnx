// sherpa-onnx/csrc/microphone.cc
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#include "sherpa-onnx/csrc/microphone.h"

#include <stdio.h>
#include <stdlib.h>

namespace sherpa_onnx {

Microphone::Microphone() {
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    exit(-1);
  }
}

Microphone::~Microphone() {
  CloseDevice();
  PaError err = Pa_Terminate();
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
  }
}

int Microphone::GetDeviceCount() const { return Pa_GetDeviceCount(); }

int Microphone::GetDefaultInputDevice() const {
  return Pa_GetDefaultInputDevice();
}

void Microphone::PrintDevices(int device_index) const {
  int num_devices = Pa_GetDeviceCount();
  fprintf(stderr, "Num devices: %d\n", num_devices);
  for (int i = 0; i != num_devices; ++i) {
    const PaDeviceInfo *info = Pa_GetDeviceInfo(i);
    fprintf(stderr, " %s %d %s\n", (i == device_index) ? "*" : " ", i,
            info->name);
  }
}

bool Microphone::OpenDevice(int index, int sample_rate, int channel,
                            PaStreamCallback cb, void *userdata) {
  if (index < 0 || index >= Pa_GetDeviceCount()) {
    fprintf(stderr, "Invalid device index: %d\n", index);
    return false;
  }

  const PaDeviceInfo *info = Pa_GetDeviceInfo(index);
  if (!info) {
    fprintf(stderr, "No device info found for index: %d\n", index);
    return false;
  }

  CloseDevice();

  fprintf(stderr, "Use device: %d\n", index);
  fprintf(stderr, "  Name: %s\n", info->name);
  fprintf(stderr, "  Max input channels: %d\n", info->maxInputChannels);

  PaStreamParameters param;
  param.device = index;
  param.channelCount = channel;
  param.sampleFormat = paFloat32;
  param.suggestedLatency = info->defaultLowInputLatency;
  param.hostApiSpecificStreamInfo = nullptr;

  PaError err =
      Pa_OpenStream(&stream, &param, nullptr, /* &outputParameters, */
                    sample_rate,
                    0,          // frames per buffer
                    paClipOff,  // we won't output out of range samples
                                // so don't bother clipping them
                    cb, userdata);
  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    return false;
  }

  err = Pa_StartStream(stream);
  fprintf(stderr, "Started\n");

  if (err != paNoError) {
    fprintf(stderr, "portaudio error: %s\n", Pa_GetErrorText(err));
    CloseDevice();
    return false;
  }
  return true;
}

void Microphone::CloseDevice() {
  if (stream) {
    PaError err = Pa_CloseStream(stream);
    if (err != paNoError) {
      fprintf(stderr, "Pa_CloseStream error: %s\n", Pa_GetErrorText(err));
    }
    stream = nullptr;
  }
}

}  // namespace sherpa_onnx
