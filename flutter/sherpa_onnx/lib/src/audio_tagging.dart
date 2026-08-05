// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

import './offline_stream.dart';
import './audio_tagging_config.dart';
import './sherpa_onnx_bindings.dart';

export './audio_tagging_config.dart';

/// Offline audio tagger.
class AudioTagging {
  AudioTagging.fromPtr({required this.ptr, required this.config});

  AudioTagging._({required this.ptr, required this.config});

  /// Create an audio tagger from [config].
  factory AudioTagging({required AudioTaggingConfig config}) {
    final c = calloc<SherpaOnnxAudioTaggingConfig>();

    final zipformerPtr = config.model.zipformer.model.toNativeUtf8();
    c.ref.model.zipformer.model = zipformerPtr;

    final cedPtr = config.model.ced.toNativeUtf8();
    c.ref.model.ced = cedPtr;

    c.ref.model.numThreads = config.model.numThreads;

    final providerPtr = config.model.provider.toNativeUtf8();
    c.ref.model.provider = providerPtr;

    c.ref.model.debug = config.model.debug ? 1 : 0;

    final labelsPtr = config.labels.toNativeUtf8();
    c.ref.labels = labelsPtr;

    if (SherpaOnnxBindings.sherpaOnnxCreateAudioTagging == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr =
        SherpaOnnxBindings.sherpaOnnxCreateAudioTagging?.call(c) ?? nullptr;

    calloc.free(labelsPtr);
    calloc.free(providerPtr);
    calloc.free(cedPtr);
    calloc.free(zipformerPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create audio tagging. Please check your config");
    }

    return AudioTagging._(ptr: ptr, config: config);
  }

  /// Release the native tagger.
  void free() {
    if (SherpaOnnxBindings.sherpaOnnxDestroyAudioTagging == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.sherpaOnnxDestroyAudioTagging?.call(ptr);
    ptr = nullptr;
  }

  /// Create an offline stream for one audio clip.
  OfflineStream createStream() {
    if (SherpaOnnxBindings.sherpaOnnxAudioTaggingCreateOfflineStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    final p = SherpaOnnxBindings.sherpaOnnxAudioTaggingCreateOfflineStream
            ?.call(ptr) ??
        nullptr;

    if (p == nullptr) {
      throw Exception("Failed to create offline stream");
    }

    return OfflineStream(ptr: p);
  }

  /// Compute the top [topK] events for [stream].
  List<AudioEvent> compute({required OfflineStream stream, required int topK}) {
    if (SherpaOnnxBindings.sherpaOnnxAudioTaggingCompute == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return <AudioEvent>[];
    }

    final pp = SherpaOnnxBindings.sherpaOnnxAudioTaggingCompute
            ?.call(ptr, stream.ptr, topK) ??
        nullptr;

    final ans = <AudioEvent>[];

    if (pp == nullptr) {
      return ans;
    }

    var i = 0;
    while (pp[i] != nullptr) {
      final p = pp[i];

      final name = p.ref.name.toDartString();
      final index = p.ref.index;
      final prob = p.ref.prob;
      final e = AudioEvent(name: name, index: index, prob: prob);
      ans.add(e);

      i += 1;
    }

    SherpaOnnxBindings.sherpaOnnxAudioTaggingFreeResults?.call(pp);

    return ans;
  }

  Pointer<SherpaOnnxAudioTagging> ptr;
  final AudioTaggingConfig config;
}
