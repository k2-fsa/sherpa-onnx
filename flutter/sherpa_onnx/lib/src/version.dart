// Copyright (c)  2025  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';
import './sherpa_onnx_bindings.dart';

/// Return the sherpa-onnx version string compiled into the native library.
String getVersion() {
  Pointer<Utf8> version = SherpaOnnxBindings.getVersionStr?.call() ?? nullptr;
  if (version == nullptr) {
    return '';
  }

  return version.toDartString();
}

/// Return the Git SHA1 of the native library build.
String getGitSha1() {
  Pointer<Utf8> gitSha1 = SherpaOnnxBindings.getGitSha1?.call() ?? nullptr;
  if (gitSha1 == nullptr) {
    return '';
  }

  return gitSha1.toDartString();
}

/// Return the Git date of the native library build.
String getGitDate() {
  Pointer<Utf8> gitDate = SherpaOnnxBindings.getGitDate?.call() ?? nullptr;
  if (gitDate == nullptr) {
    return '';
  }

  return gitDate.toDartString();
}
