// Web: create a blob URL from bytes for media_kit.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

/// Create a blob URL from bytes. Used to play user-selected files on web.
String createBlobUrl(Uint8List bytes) {
  // Create a Blob from the ArrayBuffer.
  final arrayBuffer = bytes.buffer.toJS;
  final blobCtor = globalContext.getProperty('Blob'.toJS) as JSFunction;
  // Build the parts array as a JS array containing the ArrayBuffer.
  final arrayCtor = globalContext.getProperty('Array'.toJS) as JSFunction;
  final parts = arrayCtor.callAsConstructor(arrayBuffer) as JSObject;
  final blobOptions = JSObject();
  blobOptions['type'] = 'application/octet-stream'.toJS;
  final blob = blobCtor.callAsConstructor(parts, blobOptions) as JSObject;

  // Create an object URL.
  final urlObj = globalContext.getProperty('URL'.toJS) as JSObject;
  final createUrlFn = urlObj.getProperty('createObjectURL'.toJS) as JSFunction;
  return (createUrlFn.callAsFunction(urlObj, blob) as JSString).toDart;
}
