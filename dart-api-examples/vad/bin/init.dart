import 'dart:io';
import 'dart:isolate';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

Future<void> initSherpaOnnx() async {
  var uri = await Isolate.resolvePackageUri(
      Uri.parse('package:sherpa_onnx/sherpa_onnx.dart'));

  if (uri == null) {
    print('File not found');
    exit(1);
  }

  String platform = '';

  if (Platform.isMacOS) {
    platform = 'macos';
  } else if (Platform.isLinux) {
    platform = 'linux';
  } else if (Platform.isWindows) {
    platform = 'windows';
  } else {
    throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
  }

  final libPath = p.join(p.dirname(p.fromUri(uri)), '..', platform);
  sherpa_onnx.initBindings(libPath);
}
