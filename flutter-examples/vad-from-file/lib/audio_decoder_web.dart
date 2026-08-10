// Copyright (c)  2026  Xiaomi Corporation
// Web audio decoder using Web Audio API.
import 'dart:async';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

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

    // Use Web Audio API to decode.
    final audioContextCtor =
        globalContext.getProperty('AudioContext'.toJS) as JSFunction?;

    JSFunction ctor;
    if (audioContextCtor != null) {
      ctor = audioContextCtor;
    } else {
      // Try webkitAudioContext for Safari.
      final webkitCtor =
          globalContext.getProperty('webkitAudioContext'.toJS) as JSFunction?;
      if (webkitCtor == null) return null;
      ctor = webkitCtor;
    }

    final audioContext = ctor.callAsConstructor() as JSObject;

    // Wrap the ArrayBuffer in a Uint8Array copy (avoids shared-buffer issues).
    final copiedBytes = Uint8List.fromList(bytes);
    final arrayBuffer = copiedBytes.buffer.toJS;

    final decodeFn =
        audioContext.getProperty('decodeAudioData'.toJS) as JSFunction;

    // decodeAudioData returns a Promise<AudioBuffer>.
    final promise = decodeFn.callAsFunction(audioContext, arrayBuffer) as JSPromise;
    final audioBuffer = await promise.toDart;
    if (audioBuffer == null) return null;

    final buffer = audioBuffer as JSObject;

    final sampleRate =
        (buffer.getProperty('sampleRate'.toJS) as JSNumber).toDartInt;
    final duration =
        (buffer.getProperty('duration'.toJS) as JSNumber).toDartDouble;

    // Get channel data (mono: use channel 0).
    final getChannelDataFn =
        buffer.getProperty('getChannelData'.toJS) as JSFunction;
    final channelData =
        getChannelDataFn.callAsFunction(buffer, 0.toJS) as JSFloat32Array;
    final samples = channelData.toDart;

    // Resample to 16kHz if needed.
    if (sampleRate == 16000) {
      return DecodedAudio(
        samples: Float32List.fromList(samples),
        sampleRate: 16000,
        duration: duration,
      );
    }

    // Simple linear resampling.
    final targetLength = (samples.length * 16000 / sampleRate).round();
    final resampled = Float32List(targetLength);
    for (int i = 0; i < targetLength; i++) {
      final srcIndex = i * sampleRate / 16000;
      final srcIndexInt = srcIndex.floor();
      final frac = srcIndex - srcIndexInt;
      if (srcIndexInt + 1 < samples.length) {
        resampled[i] = samples[srcIndexInt] * (1 - frac) + samples[srcIndexInt + 1] * frac;
      } else if (srcIndexInt < samples.length) {
        resampled[i] = samples[srcIndexInt];
      }
    }

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
