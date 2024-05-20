import 'dart:io';
import 'dart:ffi';

import 'src/sherpa_onnx_bindings.dart';
export 'src/speaker_identification.dart';
export 'src/online_stream.dart';

final DynamicLibrary _dylib = () {
  if (Platform.isIOS) {
    throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
  }
  if (Platform.isMacOS) {
    return DynamicLibrary.open('libsherpa-onnx-c-api.dylib');
  }
  if (Platform.isAndroid || Platform.isLinux) {
    return DynamicLibrary.open('libsherpa-onnx-c-api.so');
  }

  if (Platform.isWindows) {
    return DynamicLibrary.open('sherpa-onnx-c-api.dll');
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

void initBindings() {
  SherpaOnnxBindings.init(_dylib);
}