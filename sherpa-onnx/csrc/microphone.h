// sherpa-onnx/csrc/microphone.h
//
// Copyright (c)  2022-2023  Xiaomi Corporation

#ifndef SHERPA_ONNX_CSRC_MICROPHONE_H_
#define SHERPA_ONNX_CSRC_MICROPHONE_H_
#include "portaudio.h"  // NOLINT

namespace sherpa_onnx {

class Microphone {
  PaStream *stream = nullptr;
 public:
  Microphone();
  ~Microphone();

  int GetDeviceCount();
  int GetDefaultInputDevice();
  void PrintDevices(int sel);
  
  bool OpenDevice(int index, int samplerate, int channel, PaStreamCallback cb, void* userdata);
  void CloseDevice();
};

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_CSRC_MICROPHONE_H_
