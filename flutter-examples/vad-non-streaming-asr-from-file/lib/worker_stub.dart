// Native stub for VadAsrWorker. On native, VAD+ASR uses isolates directly
// (see vad_asr_manager.dart). This stub exists so worker_web.dart's
// conditional import compiles on native.
import 'dart:typed_data';

import './vad_asr_manager.dart' show VadAsrSegment;

typedef OnReadyCallback = void Function();
typedef OnStartedCallback = void Function();
typedef OnProgressCallback = void Function(double progress);
typedef OnSegmentCallback = void Function(VadAsrSegment segment);
typedef OnResultCallback = void Function(
    List<VadAsrSegment> segments, double elapsed, double audioDuration);
typedef OnErrorCallback = void Function(String message);

class VadAsrWorker {
  VadAsrWorker({
    required OnReadyCallback onReady,
    required OnStartedCallback onStarted,
    required OnProgressCallback onProgress,
    required OnSegmentCallback onSegment,
    required OnResultCallback onResult,
    required OnErrorCallback onError,
  });

  Future<void> init() async {}
  void runVad({
    required int runId,
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
