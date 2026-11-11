// Copyright (c)  2026  Xiaomi Corporation
// Web stub for speaker identification -- not yet implemented.
// TODO: implement using dart:js_interop and the corresponding JS wrapper.

import 'dart:typed_data';

import 'online_stream.dart';
import '../speaker_identification_config.dart';

export '../speaker_identification_config.dart';

/// Speaker embedding extractor.
///
/// Feed audio through an [OnlineStream], then call [compute] to obtain a fixed
/// dimensional embedding suitable for search or verification.
class SpeakerEmbeddingExtractor {
  SpeakerEmbeddingExtractor.fromPtr({required this.ptr, required this.dim});

  factory SpeakerEmbeddingExtractor(
      {required SpeakerEmbeddingExtractorConfig config}) {
    throw UnsupportedError(
        'SpeakerEmbeddingExtractor is not yet supported on web');
  }

  void free() {}
  OnlineStream createStream() =>
      throw UnsupportedError(
          'SpeakerEmbeddingExtractor is not yet supported on web');
  bool isReady(OnlineStream stream) => false;
  Float32List compute(OnlineStream stream) => Float32List(0);

  dynamic ptr;
  final int dim;
}

/// In-memory store of named speaker embeddings.
///
/// Use this class to add reference embeddings, search for the best matching
/// speaker, and verify whether a candidate embedding belongs to a known
/// identity.
class SpeakerEmbeddingManager {
  SpeakerEmbeddingManager.fromPtr({required this.ptr, required this.dim});

  factory SpeakerEmbeddingManager(int dim) {
    throw UnsupportedError(
        'SpeakerEmbeddingManager is not yet supported on web');
  }

  void free() {}
  bool add({required String name, required Float32List embedding}) => false;
  bool addMulti(
          {required String name, required List<Float32List> embeddingList}) =>
      false;
  bool contains(String name) => false;
  bool remove(String name) => false;
  String search({required Float32List embedding, required double threshold}) =>
      '';
  bool verify(
          {required String name,
          required Float32List embedding,
          required double threshold}) =>
      false;

  int get numSpeakers => 0;
  List<String> get allSpeakerNames => <String>[];

  dynamic ptr;
  final int dim;
}
