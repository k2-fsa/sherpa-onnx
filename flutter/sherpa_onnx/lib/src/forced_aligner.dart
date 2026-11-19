// Copyright (c)  2026  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './forced_aligner_config.dart';
import './offline_stream.dart';

export './forced_aligner_config.dart';

/// Forced aligner: aligns a transcript to an utterance and returns the
/// phone and word boundaries. See `dart-api-examples/forced-alignment-*`.
class ForcedAligner {
  ForcedAligner.fromPtr({required this.ptr, required this.config});

  ForcedAligner._({required this.ptr, required this.config});

  /// Create a forced aligner from [config].
  factory ForcedAligner({required ForcedAlignerConfig config}) {
    if (SherpaOnnxBindings.sherpaOnnxCreateForcedAligner == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final c = calloc<SherpaOnnxForcedAlignerConfig>();

    final modelDirPtr = config.modelDir.toNativeUtf8();
    c.ref.modelDir = modelDirPtr;

    final lexiconPtr = config.lexicon.toNativeUtf8();
    c.ref.lexicon = lexiconPtr;

    c.ref.options.beam = config.options.beam;
    c.ref.options.retryBeam = config.options.retryBeam;
    c.ref.options.transitionScale = config.options.transitionScale;
    c.ref.options.selfLoopScale = config.options.selfLoopScale;
    c.ref.options.acousticScale = config.options.acousticScale;

    c.ref.debug = config.debug ? 1 : 0;

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateForcedAligner?.call(c) ?? nullptr;

    calloc.free(lexiconPtr);
    calloc.free(modelDirPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create the forced aligner. Please check your config");
    }

    return ForcedAligner._(ptr: ptr, config: config);
  }

  /// Release the native forced aligner.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyForcedAligner == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.sherpaOnnxDestroyForcedAligner?.call(ptr);
    ptr = nullptr;
  }

  /// Create an utterance stream. Fill it with
  /// [OfflineStream.acceptWaveform](), then pass it to [compute]().
  ///
  /// The user has to call [OfflineStream.free]() to avoid memory leak.
  OfflineStream createStream() {
    if (SherpaOnnxBindings.sherpaOnnxForcedAlignerCreateStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final p = SherpaOnnxBindings.sherpaOnnxForcedAlignerCreateStream
            ?.call(ptr) ??
        nullptr;

    return OfflineStream(ptr: p);
  }

  /// Compute the alignment of one utterance.
  ///
  /// It returns null when the alignment itself fails. The documented empty
  /// result (empty text, no features, every word out-of-vocabulary, or word
  /// boundaries cannot be recovered) comes back non-null with empty [tiers]
  /// and an empty [textGrid].
  ForcedAlignmentResult? compute(
      {required OfflineStream stream, required String text}) {
    if (SherpaOnnxBindings.sherpaOnnxForcedAlignerCompute == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return null;
    }

    final textPtr = text.toNativeUtf8();

    final p = SherpaOnnxBindings.sherpaOnnxForcedAlignerCompute
            ?.call(ptr, stream.ptr, textPtr) ??
        nullptr;

    calloc.free(textPtr);

    if (p == nullptr) {
      return null;
    }

    try {
      // Everything is copied into Dart values before the C result is freed.
      final textGrid =
          p.ref.textGrid == nullptr ? '' : p.ref.textGrid.toDartString();

      final tiers = <AlignmentTier>[];
      final numTiers = p.ref.numTiers;
      for (var i = 0; i < numTiers; ++i) {
        final t = p.ref.tiers[i];
        final name = t.name == nullptr ? '' : t.name.toDartString();

        final intervals = <AlignmentInterval>[];
        final numIntervals = t.numIntervals;
        for (var k = 0; k < numIntervals; ++k) {
          final iv = t.intervals[k];
          intervals.add(AlignmentInterval(
            start: iv.start,
            end: iv.end,
            text: iv.text == nullptr ? '' : iv.text.toDartString(),
          ));
        }

        tiers.add(AlignmentTier(name: name, intervals: intervals));
      }

      return ForcedAlignmentResult(
        duration: p.ref.duration,
        tiers: tiers,
        textGrid: textGrid,
      );
    } finally {
      // Runs even if a UTF-8 conversion above throws.
      SherpaOnnxBindings.sherpaOnnxDestroyForcedAlignmentResult?.call(p);
    }
  }

  Pointer<SherpaOnnxForcedAligner> ptr;
  final ForcedAlignerConfig config;
}
