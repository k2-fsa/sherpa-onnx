// Native stub for web_audio.dart.
import 'dart:typed_data';

Uint8List encodeWav(Float32List samples, int sampleRate) => Uint8List(0);
void playWavBytes(Uint8List wavBytes) {}
void stopPlayback() {}
void downloadWavBytes(Uint8List wavBytes, String filename) {}
Future<void> saveWavBytesWithDialog(Uint8List wavBytes, String filename) async {}
void playAudioChunk(Float32List samples, int sampleRate) {}
void resetChunkPlayback() {}
