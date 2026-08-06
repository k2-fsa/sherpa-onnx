// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './speaker_identification.dart';
import './offline_speaker_diarization_config.dart';

export './offline_speaker_diarization_config.dart';

/// Offline speaker diarizer.
class OfflineSpeakerDiarization {
  OfflineSpeakerDiarization.fromPtr(
      {required this.ptr, required this.config, required this.sampleRate});

  OfflineSpeakerDiarization._(
      {required this.ptr, required this.config, required this.sampleRate});

  /// Release the native diarizer.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyOfflineSpeakerDiarization == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.sherpaOnnxDestroyOfflineSpeakerDiarization?.call(ptr);
    ptr = nullptr;
  }

  /// Create a diarizer from [config].
  factory OfflineSpeakerDiarization(OfflineSpeakerDiarizationConfig config) {
    if (SherpaOnnxBindings.sherpaOnnxCreateOfflineSpeakerDiarization == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxOfflineSpeakerDiarizationConfig>();

    c.ref.segmentation.pyannote.model =
        config.segmentation.pyannote.model.toNativeUtf8();
    c.ref.segmentation.numThreads = config.segmentation.numThreads;
    c.ref.segmentation.debug = config.segmentation.debug ? 1 : 0;
    c.ref.segmentation.provider = config.segmentation.provider.toNativeUtf8();

    c.ref.embedding.model = config.embedding.model.toNativeUtf8();
    c.ref.embedding.numThreads = config.embedding.numThreads;
    c.ref.embedding.debug = config.embedding.debug ? 1 : 0;
    c.ref.embedding.provider = config.embedding.provider.toNativeUtf8();

    c.ref.clustering.numClusters = config.clustering.numClusters;
    c.ref.clustering.threshold = config.clustering.threshold;

    c.ref.minDurationOn = config.minDurationOn;
    c.ref.minDurationOff = config.minDurationOff;

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateOfflineSpeakerDiarization?.call(c) ??
            nullptr;

    calloc.free(c.ref.embedding.provider);
    calloc.free(c.ref.embedding.model);
    calloc.free(c.ref.segmentation.provider);
    calloc.free(c.ref.segmentation.pyannote.model);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create offline speaker diarization. Please check your config");
    }

    int sampleRate = SherpaOnnxBindings
              .sherpaOnnxOfflineSpeakerDiarizationGetSampleRate
              ?.call(ptr) ?? 0;

    return OfflineSpeakerDiarization._(
        ptr: ptr, config: config, sampleRate: sampleRate);
  }

  /// Process a complete waveform and return speaker-labeled segments.
  List<OfflineSpeakerDiarizationSegment> process(
      {required Float32List samples}) {
    if (SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationProcess == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return <OfflineSpeakerDiarizationSegment>[];
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);
    try {
      final pList = p.asTypedList(n);
      pList.setAll(0, samples);

      final r = SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationProcess
              ?.call(ptr, p, n) ??
          nullptr;
      try {
        return _processImpl(r);
      } finally {
        if (r != nullptr) {
          SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationDestroyResult
              ?.call(r);
        }
      }
    } finally {
      calloc.free(p);
    }
  }

  List<OfflineSpeakerDiarizationSegment> processWithCallback({
    required Float32List samples,
    required int Function(int numProcessedChunks, int numTotalChunks) callback,
  }) {
    if (SherpaOnnxBindings
            .sherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg ==
        null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return <OfflineSpeakerDiarizationSegment>[];
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);
    try {
      final pList = p.asTypedList(n);
      pList.setAll(0, samples);

      final wrapper = NativeCallable<
              SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArgNative>.isolateLocal(
          (int numProcessedChunks, int numTotalChunks) {
        return callback(numProcessedChunks, numTotalChunks);
      }, exceptionalReturn: 0);
      try {
        final r = SherpaOnnxBindings
                .sherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg
                ?.call(ptr, p, n, wrapper.nativeFunction) ??
            nullptr;
        try {
          return _processImpl(r);
        } finally {
          if (r != nullptr) {
            SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationDestroyResult
                ?.call(r);
          }
        }
      } finally {
        wrapper.close();
      }
    } finally {
      calloc.free(p);
    }
  }

  List<OfflineSpeakerDiarizationSegment> _processImpl(
      Pointer<SherpaOnnxOfflineSpeakerDiarizationResult> r) {
    if (r == nullptr) {
      return <OfflineSpeakerDiarizationSegment>[];
    }

    final numSegments = SherpaOnnxBindings
            .sherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments
            ?.call(r) ??
        0;
    final segments = SherpaOnnxBindings
            .sherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime
            ?.call(r) ??
        nullptr;

    if (segments == nullptr) {
      return <OfflineSpeakerDiarizationSegment>[];
    }

    final ans = <OfflineSpeakerDiarizationSegment>[];
    for (int i = 0; i != numSegments; ++i) {
      final s = segments + i;

      final tmp = OfflineSpeakerDiarizationSegment(
          start: s.ref.start, end: s.ref.end, speaker: s.ref.speaker);
      ans.add(tmp);
    }

    SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationDestroySegment
        ?.call(segments);

    return ans;
  }

  Pointer<SherpaOnnxOfflineSpeakerDiarization> ptr;
  OfflineSpeakerDiarizationConfig config;
  final int sampleRate;
}
