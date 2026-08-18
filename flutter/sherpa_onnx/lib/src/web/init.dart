// Web platform initialization using dart:js_interop.
// The Module is set as a global JS variable by sherpa_onnx_web plugin.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

JSObject? _module;

/// Get the Emscripten Module instance.
/// Must be called after SherpaOnnxWeb.loadWasm() has completed.
JSObject getModule() {
  if (_module != null) return _module!;

  // Try to get the global Module set by sherpa_onnx_web.
  final module = globalContext.getProperty('Module'.toJS);
  if (module != null) {
    _module = module as JSObject;
    return _module!;
  }

  throw StateError(
    'WASM module not loaded. Call SherpaOnnxWeb.loadWasm() first.',
  );
}

// No-op for web — initialization is handled by SherpaOnnxWeb.
void initNativeBindings(String? path) {
  // Not used on web.
}
