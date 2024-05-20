import 'dart:ffi';
import 'package:ffi/ffi.dart';
import "./sherpa_onnx_bindings.dart";

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
  SpeakerEmbeddingExtractor._({required this.ptr});

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

    return SpeakerEmbeddingExtractor._(ptr: ptr);
  }

  void free() {
    SherpaOnnxBindings.destroySpeakerEmbeddingExtractor?.call(ptr);
    ptr = nullptr;
  }

  int get dim =>
      SherpaOnnxBindings.speakerEmbeddingExtractorDim?.call(ptr) ?? 0;

  Pointer<SherpaOnnxSpeakerEmbeddingExtractor> ptr;
}

void testSpeakerID() {
  print(SherpaOnnxBindings.createSpeakerEmbeddingExtractor);
}
