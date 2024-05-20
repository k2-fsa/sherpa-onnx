import 'dart:ffi';
import 'package:ffi/ffi.dart';

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

class SherpaOnnxBindings {
  static SherpaOnnxCreateSpeakerEmbeddingExtractor?
      createSpeakerEmbeddingExtractor;

  static SherpaOnnxDestroySpeakerEmbeddingExtractor?
      destroySpeakerEmbeddingExtractor;

  static SherpaOnnxSpeakerEmbeddingExtractorDim? speakerEmbeddingExtractorDim;

  static SherpaOnnxSpeakerEmbeddingExtractorCreateStream?
      speakerEmbeddingExtractorCreateStream;

  static DestroyOnlineStream? destroyOnlineStream;

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

    destroyOnlineStream ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOnlineStreamNative>>(
            'DestroyOnlineStream')
        .asFunction();
  }
}
