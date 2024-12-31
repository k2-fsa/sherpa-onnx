// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

import './sherpa_onnx_bindings.dart';
import './speaker_identification.dart';

class OfflineSpeakerDiarizationSegment {
  const OfflineSpeakerDiarizationSegment({
    required this.start,
    required this.end,
    required this.speaker,
  });

  @override
  String toString() {
    return 'OfflineSpeakerDiarizationSegment(start: $start, end: $end, speaker: $speaker)';
  }

  final double start;
  final double end;
  final int speaker;
}

class OfflineSpeakerSegmentationPyannoteModelConfig {
  const OfflineSpeakerSegmentationPyannoteModelConfig({
    this.model = '',
  });

  @override
  String toString() {
    return 'OfflineSpeakerSegmentationPyannoteModelConfig(model: $model)';
  }

  final String model;
}

class OfflineSpeakerSegmentationModelConfig {
  const OfflineSpeakerSegmentationModelConfig({
    this.pyannote = const OfflineSpeakerSegmentationPyannoteModelConfig(),
    this.numThreads = 1,
    this.debug = true,
    this.provider = 'cpu',
  });

  @override
  String toString() {
    return 'OfflineSpeakerSegmentationModelConfig(pyannote: $pyannote, numThreads: $numThreads, debug: $debug, provider: $provider)';
  }

  final OfflineSpeakerSegmentationPyannoteModelConfig pyannote;

  final int numThreads;
  final bool debug;
  final String provider;
}

class FastClusteringConfig {
  const FastClusteringConfig({
    this.numClusters = -1,
    this.threshold = 0.5,
  });

  @override
  String toString() {
    return 'FastClusteringConfig(numClusters: $numClusters, threshold: $threshold)';
  }

  final int numClusters;
  final double threshold;
}

class OfflineSpeakerDiarizationConfig {
  const OfflineSpeakerDiarizationConfig({
    this.segmentation = const OfflineSpeakerSegmentationModelConfig(),
    this.embedding = const SpeakerEmbeddingExtractorConfig(model: ''),
    this.clustering = const FastClusteringConfig(),
    this.minDurationOn = 0.2,
    this.minDurationOff = 0.5,
  });

  @override
  String toString() {
    return 'OfflineSpeakerDiarizationConfig(segmentation: $segmentation, embedding: $embedding, clustering: $clustering, minDurationOn: $minDurationOn, minDurationOff: $minDurationOff)';
  }

  final OfflineSpeakerSegmentationModelConfig segmentation;
  final SpeakerEmbeddingExtractorConfig embedding;
  final FastClusteringConfig clustering;
  final double minDurationOff; // in seconds
  final double minDurationOn; // in seconds
}

class OfflineSpeakerDiarization {
  OfflineSpeakerDiarization.fromPtr(
      {required this.ptr, required this.config, required this.sampleRate});

  OfflineSpeakerDiarization._(
      {required this.ptr, required this.config, required this.sampleRate});

  void free() {
    SherpaOnnxBindings.sherpaOnnxDestroyOfflineSpeakerDiarization?.call(ptr);
    ptr = nullptr;
  }

  /// The user is responsible to call the OfflineSpeakerDiarization.free()
  /// method of the returned instance to avoid memory leak.
  factory OfflineSpeakerDiarization(OfflineSpeakerDiarizationConfig config) {
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

    int sampleRate = 0;
    if (ptr != nullptr) {
      sampleRate = SherpaOnnxBindings
              .sherpaOnnxOfflineSpeakerDiarizationGetSampleRate
              ?.call(ptr) ??
          0;
    }
    return OfflineSpeakerDiarization._(
        ptr: ptr, config: config, sampleRate: sampleRate);
  }

  List<OfflineSpeakerDiarizationSegment> process(
      {required Float32List samples}) {
    if (ptr == nullptr) {
      return <OfflineSpeakerDiarizationSegment>[];
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    final r = SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationProcess
            ?.call(ptr, p, n) ??
        nullptr;

    final ans = _processImpl(r);

    SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationDestroyResult
        ?.call(r);

    return ans;
  }

  List<OfflineSpeakerDiarizationSegment> processWithCallback({
    required Float32List samples,
    required int Function(int numProcessedChunks, int numTotalChunks) callback,
  }) {
    if (ptr == nullptr) {
      return <OfflineSpeakerDiarizationSegment>[];
    }

    final n = samples.length;
    final Pointer<Float> p = calloc<Float>(n);

    final pList = p.asTypedList(n);
    pList.setAll(0, samples);

    final wrapper = NativeCallable<
            SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArgNative>.isolateLocal(
        (int numProcessedChunks, int numTotalChunks) {
      return callback(numProcessedChunks, numTotalChunks);
    }, exceptionalReturn: 0);

    final r = SherpaOnnxBindings
            .sherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg
            ?.call(ptr, p, n, wrapper.nativeFunction) ??
        nullptr;

    wrapper.close();

    final ans = _processImpl(r);

    SherpaOnnxBindings.sherpaOnnxOfflineSpeakerDiarizationDestroyResult
        ?.call(r);

    return ans;
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
