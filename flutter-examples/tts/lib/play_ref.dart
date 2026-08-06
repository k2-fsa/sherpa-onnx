// Native: write WAV bytes to a temp file and play via AudioPlayer.
import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';

Future<void> playRefWavBytes(AudioPlayer player, Uint8List wavBytes) async {
  await player.stop();
  final dir = await Directory.systemTemp.createTemp('sherpa_ref');
  final file = File('${dir.path}/ref.wav');
  await file.writeAsBytes(wavBytes);
  await player.play(DeviceFileSource(file.path));
}
