// Copyright (c)  2026  Xiaomi Corporation
// Web audio decoder using Web Audio API.
import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import './resample.dart';

/// Result of decoding an audio file.
class DecodedAudio {
  final Float32List samples;
  final int sampleRate;
  final double duration;
  const DecodedAudio({
    required this.samples,
    required this.sampleRate,
    required this.duration,
  });
}

/// Decode audio bytes to 16kHz mono Float32 PCM samples using Web Audio API.
/// Returns null if decoding fails.
Future<DecodedAudio?> decodeAudioBytes(Uint8List bytes) async {
  try {
    final audioContextCtor =
        globalContext.getProperty('AudioContext'.toJS) as JSFunction?;

    JSFunction ctor;
    if (audioContextCtor != null) {
      ctor = audioContextCtor;
    } else {
      final webkitCtor =
          globalContext.getProperty('webkitAudioContext'.toJS) as JSFunction?;
      if (webkitCtor == null) return null;
      ctor = webkitCtor;
    }

    final audioContext = ctor.callAsConstructor() as JSObject;

    // Copy bytes to avoid shared-buffer issues.
    final copiedBytes = Uint8List.fromList(bytes);
    final arrayBuffer = copiedBytes.buffer.toJS;

    final decodeFn =
        audioContext.getProperty('decodeAudioData'.toJS) as JSFunction;
    final promise = decodeFn.callAsFunction(audioContext, arrayBuffer) as JSPromise;
    final audioBuffer = await promise.toDart;
    if (audioBuffer == null) return null;

    final buffer = audioBuffer as JSObject;

    final srcSampleRate =
        (buffer.getProperty('sampleRate'.toJS) as JSNumber).toDartInt;
    final duration =
        (buffer.getProperty('duration'.toJS) as JSNumber).toDartDouble;

    // Get channel data (mono: use channel 0).
    final getChannelDataFn =
        buffer.getProperty('getChannelData'.toJS) as JSFunction;
    final channelData =
        getChannelDataFn.callAsFunction(buffer, 0.toJS) as JSFloat32Array;
    final samples = Float32List.fromList(channelData.toDart);

    // Resample to 16kHz if needed.
    final resampled = resampleTo16k(samples, srcSampleRate);

    return DecodedAudio(
      samples: resampled,
      sampleRate: 16000,
      duration: duration,
    );
  } catch (e) {
    print('Web audio decode error: $e');
    return null;
  }
}
