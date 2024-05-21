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

  static SherpaOnnxSpeakerEmbeddingExtractorIsReady?
      speakerEmbeddingExtractorIsReady;

  static SherpaOnnxCreateSpeakerEmbeddingManager? createSpeakerEmbeddingManager;

  static SherpaOnnxDestroySpeakerEmbeddingManager?
      destroySpeakerEmbeddingManager;

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

    readWave ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxReadWaveNative>>('SherpaOnnxReadWave')
        .asFunction();

    freeWave ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxFreeWaveNative>>('SherpaOnnxFreeWave')
        .asFunction();
  }
}
