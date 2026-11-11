// Web: save files via browser download.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

Future<String?> saveFileAs(String sourcePath, String destPath) async {
  return null;
}

Future<String?> saveWavBytes(Uint8List wavBytes, String suggestedName) async {
  _downloadBytes(wavBytes, suggestedName);
  return null;
}

void _downloadBytes(Uint8List bytes, String filename) {
  globalContext['_sherpaDownloadBytes'] = bytes.toJS;
  globalContext['_sherpaDownloadFilename'] = filename.toJS;

  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, '''
    (function() {
      var bytes = window._sherpaDownloadBytes;
      var name = window._sherpaDownloadFilename || 'audio.wav';
      var blob = new Blob([bytes], {type: 'audio/wav'});
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      window._sherpaDownloadBytes = null;
      window._sherpaDownloadFilename = null;
    })()
  ''' as JSAny);
}
