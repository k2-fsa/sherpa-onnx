// Web: play WAV bytes via BytesSource.
import 'dart:typed_data';
import 'package:audioplayers/audioplayers.dart';

Future<void> playWavBytes(AudioPlayer player, Uint8List wavBytes) async {
  await player.play(BytesSource(wavBytes));
}
