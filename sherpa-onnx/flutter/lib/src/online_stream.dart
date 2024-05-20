import 'dart:ffi';
import 'package:ffi/ffi.dart';
import "./sherpa_onnx_bindings.dart";

class OnlineStream {
  OnlineStream({required this.ptr});

  void free() {
    SherpaOnnxBindings.destroyOnlineStream?.call(ptr);
    ptr = nullptr;
  }

  Pointer<SherpaOnnxOnlineStream> ptr;
}
