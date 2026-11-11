// Native: delete cached WAV chunk files from temp directory.
import 'dart:io';

import 'package:path_provider/path_provider.dart';

Future<void> cleanupTempChunkFiles() async {
  try {
    final dir = await getTemporaryDirectory();
    final files = dir.listSync();
    for (final f in files) {
      if (f is File && f.path.contains('vad_seg_') && f.path.endsWith('.wav')) {
        try {
          await f.delete();
        } catch (_) {}
      }
    }
  } catch (_) {}
}
