// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';

/// Voice activity detection and buffering helpers.
///
/// See `dart-api-examples/vad/bin/vad.dart` and
/// `dart-api-examples/vad/bin/ten-vad.dart` for complete examples.
///
/// Example:
///
/// ```dart
/// final config = VadModelConfig(
///   sileroVad: const SileroVadModelConfig(
///     model: './silero_vad.onnx',
///     minSilenceDuration: 0.25,
///     minSpeechDuration: 0.5,
///   ),
///   numThreads: 1,
/// );
///
/// final vad = VoiceActivityDetector(config: config, bufferSizeInSeconds: 10);
/// final wave = readWave('./test.wav');
/// vad.acceptWaveform(wave.samples);
/// vad.flush();
/// while (!vad.isEmpty()) {
///   print(vad.front());
///   vad.pop();
/// }
/// vad.free();
/// ```

/// Silero VAD model configuration.
class SileroVadModelConfig {
  const SileroVadModelConfig(
      {this.model = '',
      this.threshold = 0.5,
      this.minSilenceDuration = 0.5,
      this.minSpeechDuration = 0.25,
      this.windowSize = 512,
      this.maxSpeechDuration = 5.0});

  factory SileroVadModelConfig.fromJson(Map<String, dynamic> json) {
    return SileroVadModelConfig(
      model: json['model'] as String? ?? '',
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      minSilenceDuration:
          (json['minSilenceDuration'] as num?)?.toDouble() ?? 0.5,
      minSpeechDuration:
          (json['minSpeechDuration'] as num?)?.toDouble() ?? 0.25,
      windowSize: json['windowSize'] as int? ?? 512,
      maxSpeechDuration: (json['maxSpeechDuration'] as num?)?.toDouble() ?? 5.0,
    );
  }

  @override
  String toString() {
    return 'SileroVadModelConfig(model: $model, threshold: $threshold, minSilenceDuration: $minSilenceDuration, minSpeechDuration: $minSpeechDuration, windowSize: $windowSize, maxSpeechDuration: $maxSpeechDuration)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'threshold': threshold,
        'minSilenceDuration': minSilenceDuration,
        'minSpeechDuration': minSpeechDuration,
        'windowSize': windowSize,
        'maxSpeechDuration': maxSpeechDuration,
      };

  final String model;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final int windowSize;
  final double maxSpeechDuration;
}

/// Ten VAD model configuration.
class TenVadModelConfig {
  const TenVadModelConfig(
      {this.model = '',
      this.threshold = 0.5,
      this.minSilenceDuration = 0.5,
      this.minSpeechDuration = 0.25,
      this.windowSize = 256,
      this.maxSpeechDuration = 5.0});

  factory TenVadModelConfig.fromJson(Map<String, dynamic> json) {
    return TenVadModelConfig(
      model: json['model'] as String? ?? '',
      threshold: (json['threshold'] as num?)?.toDouble() ?? 0.5,
      minSilenceDuration:
          (json['minSilenceDuration'] as num?)?.toDouble() ?? 0.5,
      minSpeechDuration:
          (json['minSpeechDuration'] as num?)?.toDouble() ?? 0.25,
      windowSize: json['windowSize'] as int? ?? 256,
      maxSpeechDuration: (json['maxSpeechDuration'] as num?)?.toDouble() ?? 5.0,
    );
  }

  @override
  String toString() {
    return 'TenVadModelConfig(model: $model, threshold: $threshold, minSilenceDuration: $minSilenceDuration, minSpeechDuration: $minSpeechDuration, windowSize: $windowSize, maxSpeechDuration: $maxSpeechDuration)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'threshold': threshold,
        'minSilenceDuration': minSilenceDuration,
        'minSpeechDuration': minSpeechDuration,
        'windowSize': windowSize,
        'maxSpeechDuration': maxSpeechDuration,
      };

  final String model;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final int windowSize;
  final double maxSpeechDuration;
}

/// Top-level VAD model configuration.
///
/// Configure either [sileroVad] or [tenVad] for typical use and set the shared
/// sample rate and runtime settings here.
class VadModelConfig {
  VadModelConfig({
    this.sileroVad = const SileroVadModelConfig(),
    this.sampleRate = 16000,
    this.numThreads = 1,
    this.provider = 'cpu',
    this.debug = true,
    this.tenVad = const TenVadModelConfig(),
  });

  final SileroVadModelConfig sileroVad;
  final TenVadModelConfig tenVad;
  final int sampleRate;
  final int numThreads;
  final String provider;
  final bool debug;

  factory VadModelConfig.fromJson(Map<String, dynamic> json) {
    return VadModelConfig(
      sileroVad: SileroVadModelConfig.fromJson(
          json['sileroVad'] as Map<String, dynamic>? ?? const {}),
      tenVad: TenVadModelConfig.fromJson(
          json['tenVad'] as Map<String, dynamic>? ?? const {}),
      sampleRate: json['sampleRate'] as int? ?? 16000,
      numThreads: json['numThreads'] as int? ?? 1,
      provider: json['provider'] as String? ?? 'cpu',
      debug: json['debug'] as bool? ?? true,
    );
  }

  Map<String, dynamic> toJson() => {
        'sileroVad': sileroVad.toJson(),
        'tenVad': tenVad.toJson(),
        'sampleRate': sampleRate,
        'numThreads': numThreads,
        'provider': provider,
        'debug': debug,
      };

  @override
  String toString() {
    return 'VadModelConfig(sileroVad: $sileroVad, tenVad: $tenVad, sampleRate: $sampleRate, numThreads: $numThreads, provider: $provider, debug: $debug)';
  }
}

/// One detected speech segment emitted by [VoiceActivityDetector].
class SpeechSegment {
  SpeechSegment({required this.samples, required this.start});
  final Float32List samples;
  final int start;
}

/// Circular sample buffer used by VAD-related pipelines.
class CircularBuffer {
  CircularBuffer.fromPtr({required this.ptr});

  CircularBuffer._({required this.ptr});

  /// The user has to invoke CircularBuffer.free() on the returned instance
  /// to avoid memory leak.
  factory CircularBuffer({required int capacity}) {
    assert(capacity > 0, 'capacity is $capacity');

    if (SherpaOnnxBindings.createCircularBuffer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final p =
        SherpaOnnxBindings.createCircularBuffer?.call(capacity) ?? nullptr;

    if (p == nullptr) {
      throw Exception(
          "Failed to create circular buffer. Please check your config");
    }

    return CircularBuffer._(ptr: p);
  }

  /// Release the native buffer.
  /// Release the native detector.
  void free() {
    if (SherpaOnnxBindings.destroyCircularBuffer == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyCircularBuffer?.call(ptr);
    ptr = nullptr;
  }

  /// Append samples to the tail of the buffer.
  void push(Float32List data) {
    if (SherpaOnnxBindings.circularBufferPush == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    final n = data.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, data);

    SherpaOnnxBindings.circularBufferPush?.call(ptr, p, n);

    calloc.free(p);
  }

  /// Copy [n] samples starting at [startIndex].
  Float32List get({required int startIndex, required int n}) {
    if (SherpaOnnxBindings.circularBufferGet == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return Float32List(0);
    }

    final Pointer<Float> p =
        SherpaOnnxBindings.circularBufferGet?.call(ptr, startIndex, n) ??
            nullptr;

    if (p == nullptr) {
      return Float32List(0);
    }

    final pList = p.asTypedList(n);
    final Float32List ans = Float32List.fromList(pList);

    SherpaOnnxBindings.circularBufferFree?.call(p);

    return ans;
  }

  /// Drop [n] samples from the head of the buffer.
  void pop(int n) {
    if (SherpaOnnxBindings.circularBufferPop == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.circularBufferPop?.call(ptr, n);
  }

  /// Clear the buffer contents.
  /// Reset the detector state.
  void reset() {
    if (SherpaOnnxBindings.circularBufferReset == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.circularBufferReset?.call(ptr);
  }

  int get size {
    if (SherpaOnnxBindings.circularBufferSize == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.circularBufferSize?.call(ptr) ?? 0;
  }

  int get head {
    if (SherpaOnnxBindings.circularBufferHead == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.circularBufferHead?.call(ptr) ?? 0;
  }

  Pointer<SherpaOnnxCircularBuffer> ptr;
}

/// Voice activity detector that emits [SpeechSegment] objects.
///
/// Create one with a [VadModelConfig], feed audio with [acceptWaveform], then
/// inspect queued segments with [isEmpty], [front], [pop], and [flush].
class VoiceActivityDetector {
  VoiceActivityDetector.fromPtr({required this.ptr, required this.config});

  VoiceActivityDetector._({required this.ptr, required this.config});

  // The user has to invoke VoiceActivityDetector.free() to avoid memory leak.
  /// Create a detector with an internal result buffer sized in seconds.
  factory VoiceActivityDetector(
      {required VadModelConfig config, required double bufferSizeInSeconds}) {
    if (SherpaOnnxBindings.createVoiceActivityDetector == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxVadModelConfig>();

    final sileroVadModelPtr = config.sileroVad.model.toNativeUtf8();
    c.ref.sileroVad.model = sileroVadModelPtr;

    c.ref.sileroVad.threshold = config.sileroVad.threshold;
    c.ref.sileroVad.minSilenceDuration = config.sileroVad.minSilenceDuration;
    c.ref.sileroVad.minSpeechDuration = config.sileroVad.minSpeechDuration;
    c.ref.sileroVad.windowSize = config.sileroVad.windowSize;
    c.ref.sileroVad.maxSpeechDuration = config.sileroVad.maxSpeechDuration;

    final tenVadModelPtr = config.tenVad.model.toNativeUtf8();
    c.ref.tenVad.model = tenVadModelPtr;

    c.ref.tenVad.threshold = config.tenVad.threshold;
    c.ref.tenVad.minSilenceDuration = config.tenVad.minSilenceDuration;
    c.ref.tenVad.minSpeechDuration = config.tenVad.minSpeechDuration;
    c.ref.tenVad.windowSize = config.tenVad.windowSize;
    c.ref.tenVad.maxSpeechDuration = config.tenVad.maxSpeechDuration;

    c.ref.sampleRate = config.sampleRate;
    c.ref.numThreads = config.numThreads;

    final providerPtr = config.provider.toNativeUtf8();
    c.ref.provider = providerPtr;

    c.ref.debug = config.debug ? 1 : 0;

    final ptr = SherpaOnnxBindings.createVoiceActivityDetector
            ?.call(c, bufferSizeInSeconds) ??
        nullptr;

    calloc.free(providerPtr);
    calloc.free(tenVadModelPtr);
    calloc.free(sileroVadModelPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception("Failed to create vad. Please check your config");
    }

    return VoiceActivityDetector._(ptr: ptr, config: config);
  }

  void free() {
    if (SherpaOnnxBindings.destroyVoiceActivityDetector == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroyVoiceActivityDetector?.call(ptr);
    ptr = nullptr;
  }

  /// Feed normalized waveform samples into the detector.
  void acceptWaveform(Float32List samples) {
    if (SherpaOnnxBindings.voiceActivityDetectorAcceptWaveform == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    SherpaOnnxBindings.voiceActivityDetectorAcceptWaveform?.call(ptr, p, n);

    calloc.free(p);
  }

  /// Return `true` if there are no queued speech segments.
  bool isEmpty() {
    if (SherpaOnnxBindings.voiceActivityDetectorEmpty == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return true;
    }

    final int empty =
        SherpaOnnxBindings.voiceActivityDetectorEmpty?.call(ptr) ?? 0;

    return empty == 1;
  }

  /// Return `true` if speech is currently being detected.
  bool isDetected() {
    if (SherpaOnnxBindings.voiceActivityDetectorDetected == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final int detected =
        SherpaOnnxBindings.voiceActivityDetectorDetected?.call(ptr) ?? 0;

    return detected == 1;
  }

  /// Drop the front queued speech segment.
  void pop() {
    if (SherpaOnnxBindings.voiceActivityDetectorPop == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorPop?.call(ptr);
  }

  /// Remove all queued speech segments.
  void clear() {
    if (SherpaOnnxBindings.voiceActivityDetectorClear == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorClear?.call(ptr);
  }

  /// Return the front queued speech segment.
  SpeechSegment front() {
    if (SherpaOnnxBindings.voiceActivityDetectorFront == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return SpeechSegment(samples: Float32List(0), start: 0);
    }

    final Pointer<SherpaOnnxSpeechSegment> segment =
        SherpaOnnxBindings.voiceActivityDetectorFront?.call(ptr) ?? nullptr;
    if (segment == nullptr) {
      return SpeechSegment(samples: Float32List(0), start: 0);
    }

    final sampleList = segment.ref.samples.asTypedList(segment.ref.n);
    final start = segment.ref.start;

    final samples = Float32List.fromList(sampleList);

    SherpaOnnxBindings.destroySpeechSegment?.call(segment);

    return SpeechSegment(samples: samples, start: start);
  }

  void reset() {
    if (SherpaOnnxBindings.voiceActivityDetectorReset == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorReset?.call(ptr);
  }

  /// Flush trailing buffered speech into the output queue.
  void flush() {
    if (SherpaOnnxBindings.voiceActivityDetectorFlush == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.voiceActivityDetectorFlush?.call(ptr);
  }

  Pointer<SherpaOnnxVoiceActivityDetector> ptr;
  final VadModelConfig config;
}
