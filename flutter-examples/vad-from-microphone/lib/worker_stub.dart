// Native stub for VadWorker (not used on native).
import 'dart:typed_data';

import './vad_manager.dart' show VadSegment;

typedef OnReadyCallback = void Function();
typedef OnSpeechStateChangedCallback = void Function(bool isSpeaking);
typedef OnSegmentCountChangedCallback = void Function(int count);
typedef OnSegmentDetectedCallback = void Function(VadSegment segment);
typedef OnErrorCallback = void Function(String message);

class VadWorker {
  VadWorker({
    required OnReadyCallback onReady,
    required OnSpeechStateChangedCallback onSpeechStateChanged,
    required OnSegmentCountChangedCallback onSegmentCountChanged,
    required OnSegmentDetectedCallback onSegmentDetected,
    required OnErrorCallback onError,
  });

  Future<void> init({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {}
  void acceptWaveform(Float32List samples) {}
  void dispose() {}
}
