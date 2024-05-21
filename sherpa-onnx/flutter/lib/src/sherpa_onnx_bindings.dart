// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

final class SherpaOnnxWave extends Struct {
  external Pointer<Float> samples;

  @Int32()
  external int sampleRate;

  @Int32()
  external int numSamples;
}

final class SherpaOnnxSpeakerEmbeddingExtractorConfig extends Struct {
  external Pointer<Utf8> model;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxOnlineStream extends Opaque {}

final class SherpaOnnxSpeakerEmbeddingExtractor extends Opaque {}

final class SherpaOnnxSpeakerEmbeddingManager extends Opaque {}

typedef SherpaOnnxCreateSpeakerEmbeddingManagerNative
    = Pointer<SherpaOnnxSpeakerEmbeddingManager> Function(Int32 dim);

typedef SherpaOnnxCreateSpeakerEmbeddingManager
    = Pointer<SherpaOnnxSpeakerEmbeddingManager> Function(int dim);

typedef SherpaOnnxDestroySpeakerEmbeddingManagerNative = Void Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>);

typedef SherpaOnnxDestroySpeakerEmbeddingManager = void Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>);

typedef SherpaOnnxSpeakerEmbeddingManagerAddNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>, Pointer<Float>);

typedef SherpaOnnxSpeakerEmbeddingManagerAdd = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>, Pointer<Float>);

typedef SherpaOnnxSpeakerEmbeddingManagerAddListFlattenedNative
    = Int32 Function(Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>,
        Pointer<Float>, Int32);

typedef SherpaOnnxSpeakerEmbeddingManagerAddListFlattened = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>,
    Pointer<Utf8>,
    Pointer<Float>,
    int);

typedef SherpaOnnxSpeakerEmbeddingManagerRemoveNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>);

typedef SherpaOnnxSpeakerEmbeddingManagerRemove = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>);

typedef SherpaOnnxSpeakerEmbeddingManagerContainsNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>);

typedef SherpaOnnxSpeakerEmbeddingManagerContains = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Utf8>);

typedef SherpaOnnxSpeakerEmbeddingManagerSearchNative = Pointer<Utf8> Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Float>, Float);

typedef SherpaOnnxSpeakerEmbeddingManagerSearch = Pointer<Utf8> Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>, Pointer<Float>, double);

typedef SherpaOnnxSpeakerEmbeddingManagerFreeSearchNative = Void Function(
    Pointer<Utf8>);

typedef SherpaOnnxSpeakerEmbeddingManagerFreeSearch = void Function(
    Pointer<Utf8>);

typedef SherpaOnnxSpeakerEmbeddingManagerNumSpeakersNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>);

typedef SherpaOnnxSpeakerEmbeddingManagerNumSpeakers = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>);

typedef SherpaOnnxSpeakerEmbeddingManagerVerifyNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>,
    Pointer<Utf8>,
    Pointer<Float>,
    Float);

typedef SherpaOnnxSpeakerEmbeddingManagerVerify = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingManager>,
    Pointer<Utf8>,
    Pointer<Float>,
    double);

typedef SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakersNative
    = Pointer<Pointer<Utf8>> Function(
        Pointer<SherpaOnnxSpeakerEmbeddingManager>);

typedef SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers
    = SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakersNative;

typedef SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakersNative = Void Function(
    Pointer<Pointer<Utf8>>);

typedef SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers = void Function(
    Pointer<Pointer<Utf8>>);

typedef SherpaOnnxCreateSpeakerEmbeddingExtractorNative
    = Pointer<SherpaOnnxSpeakerEmbeddingExtractor> Function(
        Pointer<SherpaOnnxSpeakerEmbeddingExtractorConfig>);

typedef SherpaOnnxCreateSpeakerEmbeddingExtractor
    = SherpaOnnxCreateSpeakerEmbeddingExtractorNative;

typedef SherpaOnnxDestroySpeakerEmbeddingExtractorNative = Void Function(
    Pointer<SherpaOnnxSpeakerEmbeddingExtractor>);

typedef SherpaOnnxDestroySpeakerEmbeddingExtractor = void Function(
    Pointer<SherpaOnnxSpeakerEmbeddingExtractor>);

typedef SherpaOnnxSpeakerEmbeddingExtractorDimNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingExtractor>);

typedef SherpaOnnxSpeakerEmbeddingExtractorDim = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingExtractor>);

typedef SherpaOnnxSpeakerEmbeddingExtractorCreateStreamNative
    = Pointer<SherpaOnnxOnlineStream> Function(
        Pointer<SherpaOnnxSpeakerEmbeddingExtractor>);

typedef SherpaOnnxSpeakerEmbeddingExtractorCreateStream
    = SherpaOnnxSpeakerEmbeddingExtractorCreateStreamNative;

typedef DestroyOnlineStreamNative = Void Function(
    Pointer<SherpaOnnxOnlineStream>);

typedef DestroyOnlineStream = void Function(Pointer<SherpaOnnxOnlineStream>);

typedef OnlineStreamAcceptWaveformNative = Void Function(
    Pointer<SherpaOnnxOnlineStream>,
    Int32 sample_rate,
    Pointer<Float>,
    Int32 n);

typedef OnlineStreamAcceptWaveform = void Function(
    Pointer<SherpaOnnxOnlineStream>, int sample_rate, Pointer<Float>, int n);

typedef OnlineStreamInputFinishedNative = Void Function(
    Pointer<SherpaOnnxOnlineStream>);

typedef OnlineStreamInputFinished = void Function(
    Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxSpeakerEmbeddingExtractorIsReadyNative = Int32 Function(
    Pointer<SherpaOnnxSpeakerEmbeddingExtractor>,
    Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxSpeakerEmbeddingExtractorIsReady = int Function(
    Pointer<SherpaOnnxSpeakerEmbeddingExtractor>,
    Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxSpeakerEmbeddingExtractorComputeEmbeddingNative
    = Pointer<Float> Function(Pointer<SherpaOnnxSpeakerEmbeddingExtractor>,
        Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding
    = SherpaOnnxSpeakerEmbeddingExtractorComputeEmbeddingNative;

typedef SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbeddingNative = Void
    Function(Pointer<Float>);

typedef SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding = void Function(
    Pointer<Float>);

typedef SherpaOnnxReadWaveNative = Pointer<SherpaOnnxWave> Function(
    Pointer<Utf8>);

typedef SherpaOnnxReadWave = SherpaOnnxReadWaveNative;

typedef SherpaOnnxFreeWaveNative = Void Function(Pointer<SherpaOnnxWave>);

typedef SherpaOnnxFreeWave = void Function(Pointer<SherpaOnnxWave>);

class SherpaOnnxBindings {
  static SherpaOnnxCreateSpeakerEmbeddingExtractor?
      createSpeakerEmbeddingExtractor;

  static SherpaOnnxDestroySpeakerEmbeddingExtractor?
      destroySpeakerEmbeddingExtractor;

  static SherpaOnnxSpeakerEmbeddingExtractorDim? speakerEmbeddingExtractorDim;

  static SherpaOnnxSpeakerEmbeddingExtractorCreateStream?
      speakerEmbeddingExtractorCreateStream;

  static SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding?
      speakerEmbeddingExtractorComputeEmbedding;

  static SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding?
      speakerEmbeddingExtractorDestroyEmbedding;

  static DestroyOnlineStream? destroyOnlineStream;

  static OnlineStreamAcceptWaveform? onlineStreamAcceptWaveform;

  static OnlineStreamInputFinished? onlineStreamInputFinished;

  static SherpaOnnxSpeakerEmbeddingExtractorIsReady?
      speakerEmbeddingExtractorIsReady;

  static SherpaOnnxCreateSpeakerEmbeddingManager? createSpeakerEmbeddingManager;

  static SherpaOnnxDestroySpeakerEmbeddingManager?
      destroySpeakerEmbeddingManager;

  static SherpaOnnxSpeakerEmbeddingManagerAdd? speakerEmbeddingManagerAdd;

  static SherpaOnnxSpeakerEmbeddingManagerAddListFlattened?
      speakerEmbeddingManagerAddListFlattened;

  static SherpaOnnxSpeakerEmbeddingManagerRemove? speakerEmbeddingManagerRemove;

  static SherpaOnnxSpeakerEmbeddingManagerContains?
      speakerEmbeddingManagerContains;

  static SherpaOnnxSpeakerEmbeddingManagerSearch? speakerEmbeddingManagerSearch;

  static SherpaOnnxSpeakerEmbeddingManagerFreeSearch?
      speakerEmbeddingManagerFreeSearch;

  static SherpaOnnxSpeakerEmbeddingManagerNumSpeakers?
      speakerEmbeddingManagerNumSpeakers;

  static SherpaOnnxSpeakerEmbeddingManagerVerify? speakerEmbeddingManagerVerify;

  static SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers?
      speakerEmbeddingManagerGetAllSpeakers;

  static SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers?
      speakerEmbeddingManagerFreeAllSpeakers;

  static SherpaOnnxReadWave? readWave;

  static SherpaOnnxFreeWave? freeWave;

  static void init(DynamicLibrary dynamicLibrary) {
    createSpeakerEmbeddingExtractor ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateSpeakerEmbeddingExtractor>>(
            'SherpaOnnxCreateSpeakerEmbeddingExtractor')
        .asFunction();

    destroySpeakerEmbeddingExtractor ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxDestroySpeakerEmbeddingExtractorNative>>(
            'SherpaOnnxDestroySpeakerEmbeddingExtractor')
        .asFunction();

    speakerEmbeddingExtractorDim ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxSpeakerEmbeddingExtractorDimNative>>(
            'SherpaOnnxSpeakerEmbeddingExtractorDim')
        .asFunction();

    speakerEmbeddingExtractorCreateStream ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingExtractorCreateStreamNative>>(
            'SherpaOnnxSpeakerEmbeddingExtractorCreateStream')
        .asFunction();

    speakerEmbeddingExtractorComputeEmbedding ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingExtractorComputeEmbeddingNative>>(
            'SherpaOnnxSpeakerEmbeddingExtractorComputeEmbedding')
        .asFunction();

    speakerEmbeddingExtractorDestroyEmbedding ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbeddingNative>>(
            'SherpaOnnxSpeakerEmbeddingExtractorDestroyEmbedding')
        .asFunction();

    destroyOnlineStream ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOnlineStreamNative>>(
            'DestroyOnlineStream')
        .asFunction();

    onlineStreamAcceptWaveform ??= dynamicLibrary
        .lookup<NativeFunction<OnlineStreamAcceptWaveformNative>>(
            'AcceptWaveform')
        .asFunction();

    onlineStreamInputFinished ??= dynamicLibrary
        .lookup<NativeFunction<OnlineStreamInputFinishedNative>>(
            'InputFinished')
        .asFunction();

    speakerEmbeddingExtractorIsReady ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingExtractorIsReadyNative>>(
            'SherpaOnnxSpeakerEmbeddingExtractorIsReady')
        .asFunction();

    createSpeakerEmbeddingManager ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateSpeakerEmbeddingManagerNative>>(
            'SherpaOnnxCreateSpeakerEmbeddingManager')
        .asFunction();

    destroySpeakerEmbeddingManager ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroySpeakerEmbeddingManagerNative>>(
            'SherpaOnnxDestroySpeakerEmbeddingManager')
        .asFunction();

    speakerEmbeddingManagerAdd ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxSpeakerEmbeddingManagerAddNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerAdd')
        .asFunction();

    speakerEmbeddingManagerAddListFlattened ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingManagerAddListFlattenedNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerAddListFlattened')
        .asFunction();

    speakerEmbeddingManagerRemove ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxSpeakerEmbeddingManagerRemoveNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerRemove')
        .asFunction();

    speakerEmbeddingManagerContains ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingManagerContainsNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerContains')
        .asFunction();

    speakerEmbeddingManagerSearch ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxSpeakerEmbeddingManagerSearchNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerSearch')
        .asFunction();

    speakerEmbeddingManagerFreeSearch ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingManagerFreeSearchNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerFreeSearch')
        .asFunction();

    speakerEmbeddingManagerNumSpeakers ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingManagerNumSpeakersNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerNumSpeakers')
        .asFunction();

    speakerEmbeddingManagerVerify ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxSpeakerEmbeddingManagerVerifyNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerVerify')
        .asFunction();

    speakerEmbeddingManagerGetAllSpeakers ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakersNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerGetAllSpeakers')
        .asFunction();

    speakerEmbeddingManagerFreeAllSpeakers ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakersNative>>(
            'SherpaOnnxSpeakerEmbeddingManagerFreeAllSpeakers')
        .asFunction();

    readWave ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxReadWaveNative>>('SherpaOnnxReadWave')
        .asFunction();

    freeWave ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxFreeWaveNative>>('SherpaOnnxFreeWave')
        .asFunction();
  }
}
