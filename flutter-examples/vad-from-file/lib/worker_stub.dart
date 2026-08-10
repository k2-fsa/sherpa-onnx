// Native stub for VadWorker (not used on native).
import 'dart:typed_data';

import './vad_manager.dart' show VadSegment;

typedef OnReadyCallback = void Function();
typedef OnProgressCallback = void Function(double progress);
typedef OnResultCallback = void Function(
    List<VadSegment> segments, double elapsed, double audioDuration);
typedef OnErrorCallback = void Function(String message);

class VadWorker {
  VadWorker({
    required OnReadyCallback onReady,
    required OnProgressCallback onProgress,
    required OnResultCallback onResult,
    required OnErrorCallback onError,
  });

  Future<void> init() async {}
  void runVad({
    required Float32List samples,
    required int sampleRate,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) {}
  void cancel() {}
  void dispose() {}
}
