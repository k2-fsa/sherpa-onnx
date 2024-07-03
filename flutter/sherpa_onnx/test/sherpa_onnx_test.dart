import 'package:flutter_test/flutter_test.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart';
import 'package:sherpa_onnx/sherpa_onnx_platform_interface.dart';
import 'package:sherpa_onnx/sherpa_onnx_method_channel.dart';
import 'package:plugin_platform_interface/plugin_platform_interface.dart';

class MockSherpaOnnxPlatform
    with MockPlatformInterfaceMixin
    implements SherpaOnnxPlatform {

  @override
  Future<String?> getPlatformVersion() => Future.value('42');
}

void main() {
  final SherpaOnnxPlatform initialPlatform = SherpaOnnxPlatform.instance;

  test('$MethodChannelSherpaOnnx is the default instance', () {
    expect(initialPlatform, isInstanceOf<MethodChannelSherpaOnnx>());
  });

  test('getPlatformVersion', () async {
    SherpaOnnx sherpaOnnxPlugin = SherpaOnnx();
    MockSherpaOnnxPlatform fakePlatform = MockSherpaOnnxPlatform();
    SherpaOnnxPlatform.instance = fakePlatform;

    expect(await sherpaOnnxPlugin.getPlatformVersion(), '42');
  });
}
