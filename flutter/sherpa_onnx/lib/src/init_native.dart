// Native platform initialization (dart:io available).
import 'dart:io';
import 'dart:ffi';

import 'sherpa_onnx_bindings.dart';

DynamicLibrary loadDylib(String? path) {
  if (Platform.isMacOS) {
    if (path == null) {
      return DynamicLibrary.open('SherpaOnnxC.framework/SherpaOnnxC');
    } else {
      return DynamicLibrary.open(
        '$path/sherpa_onnx_macos/SherpaOnnxC.xcframework/macos-arm64_x86_64/SherpaOnnxC.framework/SherpaOnnxC',
      );
    }
  }

  if (Platform.isIOS) {
    return DynamicLibrary.open('SherpaOnnxC.framework/SherpaOnnxC');
  }

  if (Platform.isAndroid || Platform.isLinux) {
    if (path == null) {
      return DynamicLibrary.open('libsherpa-onnx-c-api.so');
    } else {
      return DynamicLibrary.open('$path/libsherpa-onnx-c-api.so');
    }
  }

  if (Platform.isWindows) {
    if (path == null) {
      return DynamicLibrary.open('sherpa-onnx-c-api.dll');
    } else {
      return DynamicLibrary.open('$path\\sherpa-onnx-c-api.dll');
    }
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}

void initNativeBindings(String? path) {
  final dylib = loadDylib(path);
  SherpaOnnxBindings.init(dylib);
}
