import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:sherpa_onnx/sherpa_onnx_method_channel.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  MethodChannelSherpaOnnx platform = MethodChannelSherpaOnnx();
  const MethodChannel channel = MethodChannel('sherpa_onnx');

  setUp(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(
      channel,
      (MethodCall methodCall) async {
        return '42';
      },
    );
  });

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger.setMockMethodCallHandler(channel, null);
  });

  test('getPlatformVersion', () async {
    expect(await platform.getPlatformVersion(), '42');
  });
}
