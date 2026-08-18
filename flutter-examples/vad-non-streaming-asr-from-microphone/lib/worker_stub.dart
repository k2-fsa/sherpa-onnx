// Native stub for VadAsrMicWorker (not used on native).
import 'dart:typed_data';

import './vad_asr_manager.dart' show VadAsrSegment;

typedef OnReadyCallback = void Function();
typedef OnSpeechStateChangedCallback = void Function(bool isSpeaking);
typedef OnSegmentDetectedCallback = void Function(VadAsrSegment segment);
typedef OnErrorCallback = void Function(String message);

class VadAsrMicWorker {
  VadAsrMicWorker({
    required OnReadyCallback onReady,
    required OnSpeechStateChangedCallback onSpeechStateChanged,
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
  void reset() {}
  void dispose() {}
}
