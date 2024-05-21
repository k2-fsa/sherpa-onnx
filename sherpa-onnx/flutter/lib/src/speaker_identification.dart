import 'dart:ffi';
import 'dart:typed_data';
import 'package:ffi/ffi.dart';
import "./sherpa_onnx_bindings.dart";
import "./online_stream.dart";

class SpeakerEmbeddingExtractorConfig {
  const SpeakerEmbeddingExtractorConfig(
      {required this.model,
      this.numThreads = 1,
      this.debug = true,
      this.provider = "cpu"});

  @override
  String toString() {
    return "SpeakerEmbeddingExtractorConfig(model: $model, numThreads: $numThreads, debug: $debug, provider: $provider)";
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
    var c = calloc<SherpaOnnxSpeakerEmbeddingExtractorConfig>();

    final modelPtr = config.model.toNativeUtf8();
    c.ref.model = modelPtr;

    c.ref.numThreads = config.numThreads;
    c.ref.debug = config.debug ? 1 : 0;

    final providerPtr = config.provider.toNativeUtf8();
    c.ref.provider = providerPtr;

    final ptr =
        SherpaOnnxBindings.createSpeakerEmbeddingExtractor?.call(c) ?? nullptr;

    calloc.free(modelPtr);
    calloc.free(providerPtr);

    int dim = SherpaOnnxBindings.speakerEmbeddingExtractorDim?.call(ptr) ?? 0;

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
    int ready = SherpaOnnxBindings.speakerEmbeddingExtractorIsReady
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
  int dim;
}

class SpeakerEmbeddingManager {
  SpeakerEmbeddingManager._({required this.ptr, required this.dim});

  factory SpeakerEmbeddingManager(int dim) {
    final p =
        SherpaOnnxBindings.createSpeakerEmbeddingManager?.call(dim) ?? nullptr;
    return SpeakerEmbeddingManager._(ptr: p, dim: dim);
  }

  void free() {
    SherpaOnnxBindings.destroySpeakerEmbeddingManager?.call(this.ptr);
    this.ptr = nullptr;
  }

  Pointer<SherpaOnnxSpeakerEmbeddingManager> ptr;
  int dim;
}
