// Copyright (c)  2026  Xiaomi Corporation
// Web implementation using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'init.dart';

// Helper: call a method on the Module object with 0 args.
JSAny? _call0(String method) {
  final m = getModule();
  final fn = m.getProperty(method.toJS) as JSFunction?;
  if (fn == null) return null;
  return fn.callAsFunction(m);
}

// Helper: call a method on the Module object with 1 arg.
JSAny? _call1(String method, JSAny? arg1) {
  final m = getModule();
  final fn = m.getProperty(method.toJS) as JSFunction?;
  if (fn == null) return null;
  return fn.callAsFunction(m, arg1);
}

// Helper: convert a WASM pointer to a Dart string.
String _ptrToString(JSAny? ptr) {
  if (ptr == null) return '';
  final result = _call1('UTF8ToString', ptr);
  if (result == null) return '';
  return (result as JSString).toDart;
}

/// Return the sherpa-onnx version string compiled into the native library.
String getVersion() {
  return _ptrToString(_call0('_SherpaOnnxGetVersionStr'));
}

/// Return the Git SHA1 of the native library build.
String getGitSha1() {
  return _ptrToString(_call0('_SherpaOnnxGetGitSha1'));
}

/// Return the Git date of the native library build.
String getGitDate() {
  return _ptrToString(_call0('_SherpaOnnxGetGitDate'));
}

/// Return the onnxruntime version string used by the native library.
String getOnnxruntimeVersion() {
  return _ptrToString(_call0('_SherpaOnnxGetOnnxruntimeVersionStr'));
}
