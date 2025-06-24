// Copyright (c)  2025  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import './sherpa_onnx_bindings.dart';

String getVersion() {
  Pointer<Utf8> version = SherpaOnnxBindings.getVersionStr?.call() ?? nullptr;
  if (version == nullptr) {
    return '';
  }

  return version.toDartString();
}

String getGitSha1() {
  Pointer<Utf8> gitSha1 = SherpaOnnxBindings.getGitSha1?.call() ?? nullptr;
  if (gitSha1 == nullptr) {
    return '';
  }

  return gitSha1.toDartString();
}

String getGitDate() {
  Pointer<Utf8> gitDate = SherpaOnnxBindings.getGitDate?.call() ?? nullptr;
  if (gitDate == nullptr) {
    return '';
  }

  return gitDate.toDartString();
}
