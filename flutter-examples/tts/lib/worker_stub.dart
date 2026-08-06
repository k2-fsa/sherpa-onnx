// Native stub for TtsWorker (not used on native).
import 'dart:typed_data';
import './generated_audio.dart';

typedef OnChunkCallback = void Function(AudioChunk chunk);
typedef OnDoneCallback = void Function(GeneratedAudioItem item);
typedef OnReadyCallback = void Function(int numSpeakers);
typedef OnErrorCallback = void Function(String message);

class TtsWorker {
  TtsWorker({
    required OnChunkCallback onChunk,
    required OnDoneCallback onDone,
    required OnReadyCallback onReady,
    required OnErrorCallback onError,
  });

  Future<void> init() async {}
  void generate({
    required String text,
    int sid = 0,
    double speed = 1.0,
    int generationId = 0,
    Float32List? referenceAudio,
    int referenceSampleRate = 0,
    int numSteps = 5,
  }) {}
  void cancel() {}
  void dispose() {}
}
