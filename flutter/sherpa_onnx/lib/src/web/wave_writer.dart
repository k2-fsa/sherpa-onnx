// Copyright (c)  2026  Xiaomi Corporation
// Web stub for WAV writing -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

/// Write normalized mono PCM samples to a WAV file.
///
/// Returns `true` on success and `false` otherwise.
bool writeWave(
    {required String filename,
    required Float32List samples,
    required int sampleRate}) {
  throw UnsupportedError('writeWave is not yet supported on web');
}
