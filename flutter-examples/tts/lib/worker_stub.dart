// Native stub for TtsWorker (not used on native).
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
  void generate({required String text, int sid = 0, double speed = 1.0}) {}
  void cancel() {}
  void dispose() {}
}
