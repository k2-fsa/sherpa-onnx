// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './online_stream.dart';
import './sherpa_onnx_bindings.dart';

/// Speaker embedding extraction and speaker identification utilities.
///
/// See `dart-api-examples/speaker-identification/` for end-to-end examples.
///
/// Example:
///
/// ```dart
/// final extractor = SpeakerEmbeddingExtractor(
///   config: const SpeakerEmbeddingExtractorConfig(
///     model: './3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx',
///   ),
/// );
///
/// final stream = extractor.createStream();
/// stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);
/// while (extractor.isReady(stream)) {}
/// final embedding = extractor.compute(stream);
///
/// final manager = SpeakerEmbeddingManager(extractor.dim);
/// manager.add(name: 'alice', embedding: embedding);
/// print(manager.search(embedding: embedding, threshold: 0.6));
/// ```
class SpeakerEmbeddingExtractorConfig {
  const SpeakerEmbeddingExtractorConfig(
      {required this.model,
      this.numThreads = 1,
      this.debug = true,
      this.provider = 'cpu'});

  factory SpeakerEmbeddingExtractorConfig.fromJson(Map<String, dynamic> json) {
    return SpeakerEmbeddingExtractorConfig(
      model: json['model'] as String,
      numThreads: json['numThreads'] as int? ?? 1,
      debug: json['debug'] as bool? ?? true,
      provider: json['provider'] as String? ?? 'cpu',
    );
  }

  @override
  String toString() {
    return 'SpeakerEmbeddingExtractorConfig(model: $model, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  Map<String, dynamic> toJson() => {
        'model': model,
        'numThreads': numThreads,
        'debug': debug,
        'provider': provider,
      };

  final String model;
  final int numThreads;
  final bool debug;
  final String provider;
}

/// Speaker embedding extractor.
///
/// Feed audio through an [OnlineStream], then call [compute] to obtain a fixed
/// dimensional embedding suitable for search or verification.
class SpeakerEmbeddingExtractor {
  SpeakerEmbeddingExtractor.fromPtr({required this.ptr, required this.dim});

  SpeakerEmbeddingExtractor._({required this.ptr, required this.dim});

  /// Create an extractor from [config].
  factory SpeakerEmbeddingExtractor(
      {required SpeakerEmbeddingExtractorConfig config}) {
    final c = calloc<SherpaOnnxSpeakerEmbeddingExtractorConfig>();

    final modelPtr = config.model.toNativeUtf8();
    c.ref.model = modelPtr;

    c.ref.numThreads = config.numThreads;
    c.ref.debug = config.debug ? 1 : 0;

    final providerPtr = config.provider.toNativeUtf8();
    c.ref.provider = providerPtr;

    if (SherpaOnnxBindings.createSpeakerEmbeddingExtractor == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final ptr =
        SherpaOnnxBindings.createSpeakerEmbeddingExtractor?.call(c) ?? nullptr;

    calloc.free(providerPtr);
    calloc.free(modelPtr);
    calloc.free(c);

    if (ptr == nullptr) {
      throw Exception(
          "Failed to create speaker embedding extractor. Please check your config");
    }

    final dim = SherpaOnnxBindings.speakerEmbeddingExtractorDim?.call(ptr) ?? 0;

    return SpeakerEmbeddingExtractor._(ptr: ptr, dim: dim);
  }

  /// Release the native extractor.
  void free() {
    if (SherpaOnnxBindings.destroySpeakerEmbeddingExtractor == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroySpeakerEmbeddingExtractor?.call(ptr);
    ptr = nullptr;
  }

  /// Create an input stream for embedding extraction.
  OnlineStream createStream() {
    if (SherpaOnnxBindings.speakerEmbeddingExtractorCreateStream == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      throw Exception("Failed to create online stream");
    }

    final p =
        SherpaOnnxBindings.speakerEmbeddingExtractorCreateStream?.call(ptr) ??
            nullptr;

    if (p == nullptr) {
      throw Exception("Failed to create online stream");
    }

    return OnlineStream(ptr: p);
  }

  /// Return `true` if [stream] has enough audio for embedding extraction.
  bool isReady(OnlineStream stream) {
    if (SherpaOnnxBindings.speakerEmbeddingExtractorIsReady == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return false;
    }

    final int ready = SherpaOnnxBindings.speakerEmbeddingExtractorIsReady
            ?.call(ptr, stream.ptr) ??
        0;
    return ready == 1;
  }

  /// Compute an embedding for [stream].
  Float32List compute(OnlineStream stream) {
    if (SherpaOnnxBindings.speakerEmbeddingExtractorComputeEmbedding == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr || stream.ptr == nullptr) {
      return Float32List(0);
    }

    final Pointer<Float> embedding = SherpaOnnxBindings
            .speakerEmbeddingExtractorComputeEmbedding
            ?.call(ptr, stream.ptr) ??
        nullptr;

    if (embedding == nullptr) {
      return Float32List(0);
    }

    final embeddingList = embedding.asTypedList(dim);
    final ans = Float32List(dim);
    ans.setAll(0, embeddingList);

    SherpaOnnxBindings.speakerEmbeddingExtractorDestroyEmbedding
        ?.call(embedding);

    return ans;
  }

  Pointer<SherpaOnnxSpeakerEmbeddingExtractor> ptr;
  final int dim;
}

/// In-memory store of named speaker embeddings.
///
/// Use this class to add reference embeddings, search for the best matching
/// speaker, and verify whether a candidate embedding belongs to a known
/// identity.
class SpeakerEmbeddingManager {
  SpeakerEmbeddingManager.fromPtr({required this.ptr, required this.dim});

  SpeakerEmbeddingManager._({required this.ptr, required this.dim});

  /// Create a manager for embeddings whose dimension is [dim].
  factory SpeakerEmbeddingManager(int dim) {
    if (SherpaOnnxBindings.createSpeakerEmbeddingManager == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    final p =
        SherpaOnnxBindings.createSpeakerEmbeddingManager?.call(dim) ?? nullptr;

    if (p == nullptr) {
      throw Exception("Failed to create speaker embedding manager");
    }

    return SpeakerEmbeddingManager._(ptr: p, dim: dim);
  }

  /// Release the native manager.
  void free() {
    if (SherpaOnnxBindings.destroySpeakerEmbeddingManager == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return;
    }
    SherpaOnnxBindings.destroySpeakerEmbeddingManager?.call(ptr);
    ptr = nullptr;
  }

  /// Add one reference embedding for [name].
  bool add({required String name, required Float32List embedding}) {
    assert(embedding.length == dim, '${embedding.length} vs $dim');

    if (SherpaOnnxBindings.speakerEmbeddingManagerAdd == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final Pointer<Utf8> namePtr = name.toNativeUtf8();
    final int n = embedding.length;

    final Pointer<Float> p = calloc<Float>(n);
    final pList = p.asTypedList(n);
    pList.setAll(0, embedding);

    final int ok =
        SherpaOnnxBindings.speakerEmbeddingManagerAdd?.call(ptr, namePtr, p) ??
            0;

    calloc.free(p);
    calloc.free(namePtr);

    return ok == 1;
  }

  /// Add multiple reference embeddings for [name].
  bool addMulti(
      {required String name, required List<Float32List> embeddingList}) {
    if (SherpaOnnxBindings.speakerEmbeddingManagerAddListFlattened == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final Pointer<Utf8> namePtr = name.toNativeUtf8();
    final int n = embeddingList.length;

    final Pointer<Float> p = calloc<Float>(n * dim);
    final pList = p.asTypedList(n * dim);

    int offset = 0;
    for (final e in embeddingList) {
      assert(e.length == dim, '${e.length} vs $dim');

      pList.setAll(offset, e);
      offset += dim;
    }

    final int ok = SherpaOnnxBindings.speakerEmbeddingManagerAddListFlattened
            ?.call(ptr, namePtr, p, n) ??
        0;

    calloc.free(p);
    calloc.free(namePtr);

    return ok == 1;
  }

  /// Return `true` if [name] exists in the manager.
  bool contains(String name) {
    if (SherpaOnnxBindings.speakerEmbeddingManagerContains == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final Pointer<Utf8> namePtr = name.toNativeUtf8();

    final int found = SherpaOnnxBindings.speakerEmbeddingManagerContains
            ?.call(ptr, namePtr) ??
        0;

    calloc.free(namePtr);

    return found == 1;
  }

  /// Remove all embeddings associated with [name].
  bool remove(String name) {
    if (SherpaOnnxBindings.speakerEmbeddingManagerRemove == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final Pointer<Utf8> namePtr = name.toNativeUtf8();

    final int ok =
        SherpaOnnxBindings.speakerEmbeddingManagerRemove?.call(ptr, namePtr) ??
            0;

    calloc.free(namePtr);

    return ok == 1;
  }

  /// Search for the best matching speaker above [threshold].
  ///
  /// Returns an empty string if no speaker is found.
  String search({required Float32List embedding, required double threshold}) {
    assert(embedding.length == dim);

    if (SherpaOnnxBindings.speakerEmbeddingManagerSearch == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return '';
    }

    final Pointer<Float> p = calloc<Float>(dim);
    final pList = p.asTypedList(dim);
    pList.setAll(0, embedding);

    final Pointer<Utf8> name = SherpaOnnxBindings.speakerEmbeddingManagerSearch
            ?.call(ptr, p, threshold) ??
        nullptr;

    calloc.free(p);

    if (name == nullptr) {
      return '';
    }

    final String ans = name.toDartString();

    SherpaOnnxBindings.speakerEmbeddingManagerFreeSearch?.call(name);

    return ans;
  }

  /// Verify whether [embedding] matches [name] above [threshold].
  bool verify(
      {required String name,
       required Float32List embedding,
       required double threshold}) {
    assert(embedding.length == dim);

    if (SherpaOnnxBindings.speakerEmbeddingManagerVerify == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return false;
    }

    final Pointer<Utf8> namePtr = name.toNativeUtf8();

    final Pointer<Float> p = calloc<Float>(dim);
    final pList = p.asTypedList(dim);
    pList.setAll(0, embedding);

    final int ok = SherpaOnnxBindings.speakerEmbeddingManagerVerify
            ?.call(ptr, namePtr, p, threshold) ??
        0;

    calloc.free(p);
    calloc.free(namePtr);

    return ok == 1;
  }

  int get numSpeakers {
    if (SherpaOnnxBindings.speakerEmbeddingManagerNumSpeakers == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    if (ptr == nullptr) {
      return 0;
    }

    return SherpaOnnxBindings.speakerEmbeddingManagerNumSpeakers?.call(ptr) ??
        0;
  }

  List<String> get allSpeakerNames {
    if (SherpaOnnxBindings.speakerEmbeddingManagerGetAllSpeakers == null) {
      throw Exception("Please initialize sherpa-onnx first");
    }

    int n = numSpeakers;
    if (n == 0) {
      return <String>[];
    }

    final Pointer<Pointer<Utf8>> names =
        SherpaOnnxBindings.speakerEmbeddingManagerGetAllSpeakers?.call(ptr) ??
            nullptr;

    if (names == nullptr) {
      return <String>[];
    }

    final ans = <String>[];

    // see https://api.flutter.dev/flutter/dart-ffi/PointerPointer.html
    for (int i = 0; i != n; ++i) {
      String name = names[i].toDartString();
      ans.add(name);
    }

    SherpaOnnxBindings.speakerEmbeddingManagerFreeAllSpeakers?.call(names);

    return ans;
  }

  Pointer<SherpaOnnxSpeakerEmbeddingManager> ptr;
  final int dim;
}
