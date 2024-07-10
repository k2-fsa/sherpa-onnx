// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:ffi';

export 'src/feature_config.dart';
export 'src/offline_recognizer.dart';
export 'src/offline_stream.dart';
export 'src/online_recognizer.dart';
export 'src/online_stream.dart';
export 'src/speaker_identification.dart';
export 'src/tts.dart';
export 'src/vad.dart';
export 'src/wave_reader.dart';
export 'src/wave_writer.dart';

import 'src/sherpa_onnx_bindings.dart';

String? _path;

// see also
// https://github.com/flutter/codelabs/blob/main/ffigen_codelab/step_05/lib/ffigen_app.dart
// https://api.flutter.dev/flutter/dart-io/Platform-class.html
final DynamicLibrary _dylib = () {
  if (Platform.isMacOS || Platform.isIOS) {
    if (_path == null) {
      return DynamicLibrary.open('libsherpa-onnx-c-api.dylib');
    } else {
      return DynamicLibrary.open('$_path/libsherpa-onnx-c-api.dylib');
    }
  }

  if (Platform.isAndroid || Platform.isLinux) {
    if (_path == null) {
      return DynamicLibrary.open('libsherpa-onnx-c-api.so');
    } else {
      return DynamicLibrary.open('$_path/libsherpa-onnx-c-api.so');
    }
  }

  if (Platform.isWindows) {
    if (_path == null) {
      return DynamicLibrary.open('sherpa-onnx-c-api.dll');
    } else {
      return DynamicLibrary.open('$_path\\sherpa-onnx-c-api.dll');
    }
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

void initBindings([String? p]) {
  _path ??= p;
  SherpaOnnxBindings.init(_dylib);
}
