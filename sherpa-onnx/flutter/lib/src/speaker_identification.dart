// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';

import './online_stream.dart';
import './sherpa_onnx_bindings.dart';

class SpeakerEmbeddingExtractorConfig {
  const SpeakerEmbeddingExtractorConfig(
      {required this.model,
      this.numThreads = 1,
      this.debug = true,
      this.provider = 'cpu'});

  @override
  String toString() {
    return 'SpeakerEmbeddingExtractorConfig(model: $model, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  final String model;
  final int numThreads;
  final bool debug;
  final String provider;
}

class SpeakerEmbeddingExtractor {
  SpeakerEmbeddingExtractor._({required this.ptr, required this.dim});

  /// The user is responsible to call the SpeakerEmbeddingExtractor.free()
  /// method of the returned instance to avoid memory leak.
  factory SpeakerEmbeddingExtractor(
      {required SpeakerEmbeddingExtractorConfig config}) {
    final c = calloc<SherpaOnnxSpeakerEmbeddingExtractorConfig>();

    final modelPtr = config.model.toNativeUtf8();
    c.ref.model = modelPtr;

    c.ref.numThreads = config.numThreads;
    c.ref.debug = config.debug ? 1 : 0;

    final providerPtr = config.provider.toNativeUtf8();
    c.ref.provider = providerPtr;

    final ptr =
        SherpaOnnxBindings.createSpeakerEmbeddingExtractor?.call(c) ?? nullptr;

    calloc.free(providerPtr);
    calloc.free(modelPtr);
    calloc.free(c);

    final dim = SherpaOnnxBindings.speakerEmbeddingExtractorDim?.call(ptr) ?? 0;

    return SpeakerEmbeddingExtractor._(ptr: ptr, dim: dim);
  }

  void free() {
    SherpaOnnxBindings.destroySpeakerEmbeddingExtractor?.call(ptr);
    ptr = nullptr;
  }

  /// The user has to invoke stream.free() on the returned instance
  /// to avoid memory leak
  OnlineStream createStream() {
    final p =
        SherpaOnnxBindings.speakerEmbeddingExtractorCreateStream?.call(ptr) ??
            nullptr;

    return OnlineStream(ptr: p);
  }

  bool isReady(OnlineStream stream) {
    final int ready = SherpaOnnxBindings.speakerEmbeddingExtractorIsReady
            ?.call(this.ptr, stream.ptr) ??
        0;
    return ready == 1;
  }

  Float32List compute(OnlineStream stream) {
    final Pointer<Float> embedding = SherpaOnnxBindings
            .speakerEmbeddingExtractorComputeEmbedding
            ?.call(this.ptr, stream.ptr) ??
        nullptr;

    if (embedding == nullptr) {
      return Float32List(0);
    }

    final embeddingList = embedding.asTypedList(this.dim);
    final ans = Float32List(this.dim);
    ans.setAll(0, embeddingList);

    SherpaOnnxBindings.speakerEmbeddingExtractorDestroyEmbedding
        ?.call(embedding);

    return ans;
  }

  Pointer<SherpaOnnxSpeakerEmbeddingExtractor> ptr;
  final int dim;
}

class SpeakerEmbeddingManager {
  SpeakerEmbeddingManager._({required this.ptr, required this.dim});

  // The user has to use SpeakerEmbeddingManager.free() to avoid memory leak
  factory SpeakerEmbeddingManager(int dim) {
    final p =
        SherpaOnnxBindings.createSpeakerEmbeddingManager?.call(dim) ?? nullptr;
    return SpeakerEmbeddingManager._(ptr: p, dim: dim);
  }

  void free() {
    SherpaOnnxBindings.destroySpeakerEmbeddingManager?.call(this.ptr);
    this.ptr = nullptr;
  }

  /// Return true if added successfully; return false otherwise
  bool add({required String name, required Float32List embedding}) {
    assert(embedding.length == this.dim, '${embedding.length} vs ${this.dim}');

    final Pointer<Utf8> namePtr = name.toNativeUtf8();
    final int n = embedding.length;

    final Pointer<Float> p = calloc<Float>(n);
    final pList = p.asTypedList(n);
    pList.setAll(0, embedding);

    final int ok = SherpaOnnxBindings.speakerEmbeddingManagerAdd
            ?.call(this.ptr, namePtr, p) ??
        0;

    calloc.free(p);
    calloc.free(namePtr);

    return ok == 1;
  }

  bool addMulti(
      {required String name, required List<Float32List> embeddingList}) {
    final Pointer<Utf8> namePtr = name.toNativeUtf8();
    final int n = embeddingList.length;

    final Pointer<Float> p = calloc<Float>(n * this.dim);
    final pList = p.asTypedList(n * this.dim);

    int offset = 0;
    for (final e in embeddingList) {
      assert(e.length == this.dim, '${e.length} vs ${this.dim}');

      pList.setAll(offset, e);
      offset += this.dim;
    }

    final int ok = SherpaOnnxBindings.speakerEmbeddingManagerAddListFlattened
            ?.call(this.ptr, namePtr, p, n) ??
        0;

    calloc.free(p);
    calloc.free(namePtr);

    return ok == 1;
  }

  bool contains(String name) {
    final Pointer<Utf8> namePtr = name.toNativeUtf8();

    final int found = SherpaOnnxBindings.speakerEmbeddingManagerContains
            ?.call(this.ptr, namePtr) ??
        0;

    calloc.free(namePtr);

    return found == 1;
  }

  bool remove(String name) {
    final Pointer<Utf8> namePtr = name.toNativeUtf8();

    final int ok = SherpaOnnxBindings.speakerEmbeddingManagerRemove
            ?.call(this.ptr, namePtr) ??
        0;

    calloc.free(namePtr);

    return ok == 1;
  }

  /// Return an empty string if no speaker is found
  String search({required Float32List embedding, required double threshold}) {
    assert(embedding.length == this.dim);

    final Pointer<Float> p = calloc<Float>(this.dim);
    final pList = p.asTypedList(this.dim);
    pList.setAll(0, embedding);

    final Pointer<Utf8> name = SherpaOnnxBindings.speakerEmbeddingManagerSearch
            ?.call(this.ptr, p, threshold) ??
        nullptr;

    calloc.free(p);

    if (name == nullptr) {
      return '';
    }

    final String ans = name.toDartString();

    SherpaOnnxBindings.speakerEmbeddingManagerFreeSearch?.call(name);

    return ans;
  }

  bool verify(
      {required String name,
      required Float32List embedding,
      required double threshold}) {
    assert(embedding.length == this.dim);

    final Pointer<Utf8> namePtr = name.toNativeUtf8();

    final Pointer<Float> p = calloc<Float>(this.dim);
    final pList = p.asTypedList(this.dim);
    pList.setAll(0, embedding);

    final int ok = SherpaOnnxBindings.speakerEmbeddingManagerVerify
            ?.call(this.ptr, namePtr, p, threshold) ??
        0;

    calloc.free(p);
    calloc.free(namePtr);

    return ok == 1;
  }

  int get numSpeakers =>
      SherpaOnnxBindings.speakerEmbeddingManagerNumSpeakers?.call(this.ptr) ??
      0;

  List<String> get allSpeakerNames {
    int n = this.numSpeakers;
    if (n == 0) {
      return <String>[];
    }

    final Pointer<Pointer<Utf8>> names = SherpaOnnxBindings
            .speakerEmbeddingManagerGetAllSpeakers
            ?.call(this.ptr) ??
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
