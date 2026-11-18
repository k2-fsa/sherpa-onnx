// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of ForcedAligner using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import '../forced_aligner_config.dart';
import 'init.dart';
import 'offline_stream.dart';

export '../forced_aligner_config.dart';

/// Forced aligner (web implementation).
///
/// Aligns a transcript to an utterance and returns the phone and word
/// boundaries. See `dart-api-examples/forced-alignment-*`.
class ForcedAligner {
  ForcedAligner.fromPtr({required this.ptr, required this.config});

  ForcedAligner._({required this.ptr, required this.config});

  /// Create a forced aligner from [config].
  factory ForcedAligner({required ForcedAlignerConfig config}) {
    final m = getModule();

    // Use the ForcedAligner class from sherpa-onnx-forced-aligner.js.
    final ctor = globalContext.getProperty('ForcedAligner'.toJS) as JSFunction?;
    if (ctor == null) {
      throw StateError(
          'ForcedAligner not found. Is sherpa-onnx-forced-aligner.js loaded?');
    }

    // Build config JSON matching JS wrapper expectations.
    final jsConfig = _buildConfig(config);

    final handle = ctor.callAsConstructor(jsConfig, m);
    if (handle == null) {
      throw Exception('Failed to create the forced aligner');
    }

    return ForcedAligner._(ptr: handle, config: config);
  }

  /// Release the native forced aligner.
  void free() {
    if (_freed) return;
    final handle = ptr as JSObject;
    final freeFn = handle.getProperty('free'.toJS) as JSFunction?;
    freeFn?.callAsFunction(handle);
    _freed = true;
  }

  /// Create an utterance stream. Fill it with
  /// [OfflineStream.acceptWaveform](), then pass it to [compute]().
  ///
  /// The user has to call [OfflineStream.free]() to avoid memory leak.
  OfflineStream createStream() {
    final handle = ptr as JSObject;
    final createStreamFn =
        handle.getProperty('createStream'.toJS) as JSFunction?;
    final streamHandle = createStreamFn?.callAsFunction(handle);
    if (streamHandle == null) {
      throw Exception('Failed to create OfflineStream');
    }
    return OfflineStream(ptr: streamHandle);
  }

  /// Compute the alignment of one utterance.
  ///
  /// On web a failed alignment is reported as an empty result (the JS glue
  /// cannot distinguish it from the documented empty result); native
  /// returns null in that case. The documented empty result (empty text,
  /// no features, every word out-of-vocabulary, or word boundaries cannot
  /// be recovered) comes back with empty [tiers] and an empty [textGrid].
  ForcedAlignmentResult? compute(
      {required OfflineStream stream, required String text}) {
    final handle = ptr as JSObject;
    final computeFn = handle.getProperty('compute'.toJS) as JSFunction?;
    if (computeFn == null) {
      return null;
    }

    final jsResult =
        computeFn.callAsFunction(handle, stream.ptr as JSObject, text.toJS);
    if (jsResult == null) {
      return null;
    }

    final result = jsResult as JSObject;

    // The JS glue spells the key "textgrid".
    final textGrid =
        (result.getProperty('textgrid'.toJS) as JSString?)?.toDart ?? '';

    final tiers = <AlignmentTier>[];
    final tiersValue = result.getProperty('tiers'.toJS);
    if (tiersValue != null) {
      for (final t in (tiersValue as JSArray).toDart) {
        final tier = t as JSObject;

        final intervals = <AlignmentInterval>[];
        final intervalsValue = tier.getProperty('intervals'.toJS);
        if (intervalsValue != null) {
          for (final i in (intervalsValue as JSArray).toDart) {
            final iv = i as JSObject;
            intervals.add(AlignmentInterval(
              start:
                  (iv.getProperty('start'.toJS) as JSNumber?)?.toDartDouble ??
                      0,
              end: (iv.getProperty('end'.toJS) as JSNumber?)?.toDartDouble ??
                  0,
              text: (iv.getProperty('text'.toJS) as JSString?)?.toDart ?? '',
            ));
          }
        }

        tiers.add(AlignmentTier(
          name: (tier.getProperty('name'.toJS) as JSString?)?.toDart ?? '',
          intervals: intervals,
        ));
      }
    }

    return ForcedAlignmentResult(
      duration:
          (result.getProperty('duration'.toJS) as JSNumber?)?.toDartDouble ??
              0,
      tiers: tiers,
      textGrid: textGrid,
    );
  }

  static JSObject _buildConfig(ForcedAlignerConfig config) {
    final jsConfig = JSObject();
    jsConfig['modelDir'] = config.modelDir.toJS;
    jsConfig['lexicon'] = config.lexicon.toJS;

    final options = JSObject();
    options['beam'] = config.options.beam.toJS;
    options['retryBeam'] = config.options.retryBeam.toJS;
    options['transitionScale'] = config.options.transitionScale.toJS;
    options['selfLoopScale'] = config.options.selfLoopScale.toJS;
    options['acousticScale'] = config.options.acousticScale.toJS;
    jsConfig['options'] = options;

    jsConfig['debug'] = config.debug.toJS;

    return jsConfig;
  }

  final dynamic ptr;
  final ForcedAlignerConfig config;
  bool _freed = false;
}
