import 'package:plugin_platform_interface/plugin_platform_interface.dart';

import 'sherpa_onnx_method_channel.dart';

abstract class SherpaOnnxPlatform extends PlatformInterface {
  /// Constructs a SherpaOnnxPlatform.
  SherpaOnnxPlatform() : super(token: _token);

  static final Object _token = Object();

  static SherpaOnnxPlatform _instance = MethodChannelSherpaOnnx();

  /// The default instance of [SherpaOnnxPlatform] to use.
  ///
  /// Defaults to [MethodChannelSherpaOnnx].
  static SherpaOnnxPlatform get instance => _instance;

  /// Platform-specific implementations should set this with their own
  /// platform-specific class that extends [SherpaOnnxPlatform] when
  /// they register themselves.
  static set instance(SherpaOnnxPlatform instance) {
    PlatformInterface.verifyToken(instance, _token);
    _instance = instance;
  }

  Future<String?> getPlatformVersion() {
    throw UnimplementedError('platformVersion() has not been implemented.');
  }
}
