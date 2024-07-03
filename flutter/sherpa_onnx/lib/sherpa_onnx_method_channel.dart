import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import 'sherpa_onnx_platform_interface.dart';

/// An implementation of [SherpaOnnxPlatform] that uses method channels.
class MethodChannelSherpaOnnx extends SherpaOnnxPlatform {
  /// The method channel used to interact with the native platform.
  @visibleForTesting
  final methodChannel = const MethodChannel('sherpa_onnx');

  @override
  Future<String?> getPlatformVersion() async {
    final version = await methodChannel.invokeMethod<String>('getPlatformVersion');
    return version;
  }
}
