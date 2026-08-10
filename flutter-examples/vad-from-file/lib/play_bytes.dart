// Native: write WAV bytes to temp file and play via AudioPlayer.
import 'dart:io';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:path_provider/path_provider.dart';

Future<void> playWavBytes(AudioPlayer player, Uint8List wavBytes) async {
  final dir = await getTemporaryDirectory();
  await dir.create(recursive: true);
  final file =
      File('${dir.path}/vad_seg_${DateTime.now().microsecondsSinceEpoch}.wav');
  await file.writeAsBytes(wavBytes);
  // play() automatically stops any previous playback.
  await player.play(DeviceFileSource(file.path));
}
