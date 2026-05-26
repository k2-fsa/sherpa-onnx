// sherpa-onnx/jni/speech-denoiser.h
//
// Copyright (c)  2026  Xiaomi Corporation

#ifndef SHERPA_ONNX_JNI_SPEECH_DENOISER_H_
#define SHERPA_ONNX_JNI_SPEECH_DENOISER_H_

#include "sherpa-onnx/csrc/offline-speech-denoiser.h"
#include "sherpa-onnx/csrc/online-speech-denoiser.h"
#include "sherpa-onnx/jni/common.h"

namespace sherpa_onnx {

OfflineSpeechDenoiserModelConfig GetOfflineSpeechDenoiserModelConfig(
    JNIEnv *env, jobject model, bool *ok);

OfflineSpeechDenoiserConfig GetOfflineSpeechDenoiserConfig(
    JNIEnv *env, jobject config, bool *ok);

OnlineSpeechDenoiserConfig GetOnlineSpeechDenoiserConfig(
    JNIEnv *env, jobject config, bool *ok);

jobject NewDenoisedAudio(JNIEnv *env, const DenoisedAudio &denoised);

}  // namespace sherpa_onnx

#endif  // SHERPA_ONNX_JNI_SPEECH_DENOISER_H_
