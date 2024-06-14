// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

final class SherpaOnnxFeatureConfig extends Struct {
  @Int32()
  external int sampleRate;

  @Int32()
  external int featureDim;
}

final class SherpaOnnxOfflineTransducerModelConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;
  external Pointer<Utf8> joiner;
}

final class SherpaOnnxOfflineParaformerModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineNemoEncDecCtcModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineWhisperModelConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;
  external Pointer<Utf8> language;
  external Pointer<Utf8> task;

  @Int32()
  external int tailPaddings;
}

final class SherpaOnnxOfflineTdnnModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineLMConfig extends Struct {
  external Pointer<Utf8> model;

  @Float()
  external double scale;
}

final class SherpaOnnxOfflineModelConfig extends Struct {
  external SherpaOnnxOfflineTransducerModelConfig transducer;
  external SherpaOnnxOfflineParaformerModelConfig paraformer;
  external SherpaOnnxOfflineNemoEncDecCtcModelConfig nemoCtc;
  external SherpaOnnxOfflineWhisperModelConfig whisper;
  external SherpaOnnxOfflineTdnnModelConfig tdnn;

  external Pointer<Utf8> tokens;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;

  external Pointer<Utf8> modelType;
}

final class SherpaOnnxOfflineRecognizerConfig extends Struct {
  external SherpaOnnxFeatureConfig feat;
  external SherpaOnnxOfflineModelConfig model;
  external SherpaOnnxOfflineLMConfig lm;
  external Pointer<Utf8> decodingMethod;

  @Int32()
  external int maxActivePaths;

  external Pointer<Utf8> hotwordsFile;

  @Float()
  external double hotwordsScore;
}

final class SherpaOnnxOnlineTransducerModelConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;
  external Pointer<Utf8> joiner;
}

final class SherpaOnnxOnlineParaformerModelConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;
}

final class SherpaOnnxOnlineZipformer2CtcModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOnlineModelConfig extends Struct {
  external SherpaOnnxOnlineTransducerModelConfig transducer;
  external SherpaOnnxOnlineParaformerModelConfig paraformer;
  external SherpaOnnxOnlineZipformer2CtcModelConfig zipformer2Ctc;

  external Pointer<Utf8> tokens;

  @Int32()
  external int numThreads;

  external Pointer<Utf8> provider;

  @Int32()
  external int debug;

  external Pointer<Utf8> modelType;
}

final class SherpaOnnxOnlineCtcFstDecoderConfig extends Struct {
  external Pointer<Utf8> graph;

  @Int32()
  external int maxActive;
}

final class SherpaOnnxOnlineRecognizerConfig extends Struct {
  external SherpaOnnxFeatureConfig feat;
  external SherpaOnnxOnlineModelConfig model;
  external Pointer<Utf8> decodingMethod;

  @Int32()
  external int maxActivePaths;

  @Int32()
  external int enableEndpoint;

  @Float()
  external double rule1MinTrailingSilence;

  @Float()
  external double rule2MinTrailingSilence;

  @Float()
  external double rule3MinUtteranceLength;

  external Pointer<Utf8> hotwordsFile;

  @Float()
  external double hotwordsScore;

  external SherpaOnnxOnlineCtcFstDecoderConfig ctcFstDecoderConfig;
}

final class SherpaOnnxSileroVadModelConfig extends Struct {
  external Pointer<Utf8> model;

  @Float()
  external double threshold;

  @Float()
  external double minSilenceDuration;

  @Float()
  external double minSpeechDuration;

  @Int32()
  external int windowSize;
}

final class SherpaOnnxVadModelConfig extends Struct {
  external SherpaOnnxSileroVadModelConfig sileroVad;

  @Int32()
  external int sampleRate;

  @Int32()
  external int numThreads;

  external Pointer<Utf8> provider;

  @Int32()
  external int debug;
}

final class SherpaOnnxSpeechSegment extends Struct {
  @Int32()
  external int start;

  external Pointer<Float> samples;

  @Int32()
  external int n;
}

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

final class SherpaOnnxCircularBuffer extends Opaque {}

final class SherpaOnnxVoiceActivityDetector extends Opaque {}

final class SherpaOnnxOnlineStream extends Opaque {}

final class SherpaOnnxOnlineRecognizer extends Opaque {}

final class SherpaOnnxOfflineRecognizer extends Opaque {}

final class SherpaOnnxOfflineStream extends Opaque {}

final class SherpaOnnxSpeakerEmbeddingExtractor extends Opaque {}

final class SherpaOnnxSpeakerEmbeddingManager extends Opaque {}

typedef CreateOfflineRecognizerNative = Pointer<SherpaOnnxOfflineRecognizer>
    Function(Pointer<SherpaOnnxOfflineRecognizerConfig>);

typedef CreateOfflineRecognizer = CreateOfflineRecognizerNative;

typedef DestroyOfflineRecognizerNative = Void Function(
    Pointer<SherpaOnnxOfflineRecognizer>);

typedef DestroyOfflineRecognizer = void Function(
    Pointer<SherpaOnnxOfflineRecognizer>);

typedef CreateOfflineStreamNative = Pointer<SherpaOnnxOfflineStream> Function(
    Pointer<SherpaOnnxOfflineRecognizer>);

typedef CreateOfflineStream = CreateOfflineStreamNative;

typedef DestroyOfflineStreamNative = Void Function(
    Pointer<SherpaOnnxOfflineStream>);

typedef DestroyOfflineStream = void Function(Pointer<SherpaOnnxOfflineStream>);

typedef AcceptWaveformOfflineNative = Void Function(
    Pointer<SherpaOnnxOfflineStream>, Int32, Pointer<Float>, Int32);

typedef AcceptWaveformOffline = void Function(
    Pointer<SherpaOnnxOfflineStream>, int, Pointer<Float>, int);

typedef DecodeOfflineStreamNative = Void Function(
    Pointer<SherpaOnnxOfflineRecognizer>, Pointer<SherpaOnnxOfflineStream>);

typedef DecodeOfflineStream = void Function(
    Pointer<SherpaOnnxOfflineRecognizer>, Pointer<SherpaOnnxOfflineStream>);

typedef GetOfflineStreamResultAsJsonNative = Pointer<Utf8> Function(
    Pointer<SherpaOnnxOfflineStream>);

typedef GetOfflineStreamResultAsJson = GetOfflineStreamResultAsJsonNative;

typedef DestroyOfflineStreamResultJsonNative = Void Function(Pointer<Utf8>);

typedef DestroyOfflineStreamResultJson = void Function(Pointer<Utf8>);

typedef CreateOnlineRecognizerNative = Pointer<SherpaOnnxOnlineRecognizer>
    Function(Pointer<SherpaOnnxOnlineRecognizerConfig>);

typedef CreateOnlineRecognizer = CreateOnlineRecognizerNative;

typedef DestroyOnlineRecognizerNative = Void Function(
    Pointer<SherpaOnnxOnlineRecognizer>);

typedef DestroyOnlineRecognizer = void Function(
    Pointer<SherpaOnnxOnlineRecognizer>);

typedef CreateOnlineStreamNative = Pointer<SherpaOnnxOnlineStream> Function(
    Pointer<SherpaOnnxOnlineRecognizer>);

typedef CreateOnlineStream = CreateOnlineStreamNative;

typedef CreateOnlineStreamWithHotwordsNative = Pointer<SherpaOnnxOnlineStream>
    Function(Pointer<SherpaOnnxOnlineRecognizer>, Pointer<Utf8>);

typedef CreateOnlineStreamWithHotwords = CreateOnlineStreamWithHotwordsNative;

typedef IsOnlineStreamReadyNative = Int32 Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef IsOnlineStreamReady = int Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef DecodeOnlineStreamNative = Void Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef DecodeOnlineStream = void Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef GetOnlineStreamResultAsJsonNative = Pointer<Utf8> Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef GetOnlineStreamResultAsJson = GetOnlineStreamResultAsJsonNative;

typedef ResetNative = Void Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef Reset = void Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef IsEndpointNative = Int32 Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef IsEndpoint = int Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef DestroyOnlineStreamResultJsonNative = Void Function(Pointer<Utf8>);

typedef DestroyOnlineStreamResultJson = void Function(Pointer<Utf8>);

typedef SherpaOnnxCreateVoiceActivityDetectorNative
    = Pointer<SherpaOnnxVoiceActivityDetector> Function(
        Pointer<SherpaOnnxVadModelConfig>, Float);

typedef SherpaOnnxCreateVoiceActivityDetector
    = Pointer<SherpaOnnxVoiceActivityDetector> Function(
        Pointer<SherpaOnnxVadModelConfig>, double);

typedef SherpaOnnxDestroyVoiceActivityDetectorNative = Void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxDestroyVoiceActivityDetector = void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorAcceptWaveformNative = Void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>, Pointer<Float>, Int32);

typedef SherpaOnnxVoiceActivityDetectorAcceptWaveform = void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>, Pointer<Float>, int);

typedef SherpaOnnxVoiceActivityDetectorEmptyNative = Int32 Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorEmpty = int Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorDetectedNative = Int32 Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorDetected = int Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorPopNative = Void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorPop = void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorClearNative = Void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorClear = void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorResetNative = Void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorReset = void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorFrontNative
    = Pointer<SherpaOnnxSpeechSegment> Function(
        Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorFront
    = SherpaOnnxVoiceActivityDetectorFrontNative;

typedef SherpaOnnxDestroySpeechSegmentNative = Void Function(
    Pointer<SherpaOnnxSpeechSegment>);

typedef SherpaOnnxDestroySpeechSegment = void Function(
    Pointer<SherpaOnnxSpeechSegment>);

typedef SherpaOnnxCreateCircularBufferNative = Pointer<SherpaOnnxCircularBuffer>
    Function(Int32);

typedef SherpaOnnxCreateCircularBuffer = Pointer<SherpaOnnxCircularBuffer>
    Function(int);

typedef SherpaOnnxDestroyCircularBufferNative = Void Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxDestroyCircularBuffer = void Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCircularBufferPushNative = Void Function(
    Pointer<SherpaOnnxCircularBuffer>, Pointer<Float>, Int32);

typedef SherpaOnnxCircularBufferPush = void Function(
    Pointer<SherpaOnnxCircularBuffer>, Pointer<Float>, int);

typedef SherpaOnnxCircularBufferGetNative = Pointer<Float> Function(
    Pointer<SherpaOnnxCircularBuffer>, Int32, Int32);

typedef SherpaOnnxCircularBufferGet = Pointer<Float> Function(
    Pointer<SherpaOnnxCircularBuffer>, int, int);

typedef SherpaOnnxCircularBufferFreeNative = Void Function(Pointer<Float>);

typedef SherpaOnnxCircularBufferFree = void Function(Pointer<Float>);

typedef SherpaOnnxCircularBufferPopNative = Void Function(
    Pointer<SherpaOnnxCircularBuffer>, Int32);

typedef SherpaOnnxCircularBufferPop = void Function(
    Pointer<SherpaOnnxCircularBuffer>, int);

typedef SherpaOnnxCircularBufferSizeNative = Int32 Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCircularBufferSize = int Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCircularBufferHeadNative = Int32 Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCircularBufferHead = int Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCircularBufferResetNative = Void Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCircularBufferReset = void Function(
    Pointer<SherpaOnnxCircularBuffer>);

typedef SherpaOnnxCreateSpeakerEmbeddingManagerNative
    = Pointer<SherpaOnnxSpeakerEmbeddingManager> Function(Int32);

typedef SherpaOnnxCreateSpeakerEmbeddingManager
    = Pointer<SherpaOnnxSpeakerEmbeddingManager> Function(int);

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
    Pointer<SherpaOnnxOnlineStream>, Int32, Pointer<Float>, Int32);

typedef OnlineStreamAcceptWaveform = void Function(
    Pointer<SherpaOnnxOnlineStream>, int, Pointer<Float>, int);

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

typedef SherpaOnnxWriteWaveNative = Int32 Function(
    Pointer<Float>, Int32, Int32, Pointer<Utf8>);

typedef SherpaOnnxWriteWave = int Function(
    Pointer<Float>, int, int, Pointer<Utf8>);

typedef SherpaOnnxFreeWaveNative = Void Function(Pointer<SherpaOnnxWave>);

typedef SherpaOnnxFreeWave = void Function(Pointer<SherpaOnnxWave>);

class SherpaOnnxBindings {
  static CreateOfflineRecognizer? createOfflineRecognizer;
  static DestroyOfflineRecognizer? destroyOfflineRecognizer;
  static CreateOfflineStream? createOfflineStream;
  static DestroyOfflineStream? destroyOfflineStream;
  static AcceptWaveformOffline? acceptWaveformOffline;
  static DecodeOfflineStream? decodeOfflineStream;
  static GetOfflineStreamResultAsJson? getOfflineStreamResultAsJson;
  static DestroyOfflineStreamResultJson? destroyOfflineStreamResultJson;

  static CreateOnlineRecognizer? createOnlineRecognizer;

  static DestroyOnlineRecognizer? destroyOnlineRecognizer;

  static CreateOnlineStream? createOnlineStream;

  static CreateOnlineStreamWithHotwords? createOnlineStreamWithHotwords;

  static IsOnlineStreamReady? isOnlineStreamReady;

  static DecodeOnlineStream? decodeOnlineStream;

  static GetOnlineStreamResultAsJson? getOnlineStreamResultAsJson;

  static Reset? reset;

  static IsEndpoint? isEndpoint;

  static DestroyOnlineStreamResultJson? destroyOnlineStreamResultJson;

  static SherpaOnnxCreateVoiceActivityDetector? createVoiceActivityDetector;

  static SherpaOnnxDestroyVoiceActivityDetector? destroyVoiceActivityDetector;

  static SherpaOnnxVoiceActivityDetectorAcceptWaveform?
      voiceActivityDetectorAcceptWaveform;

  static SherpaOnnxVoiceActivityDetectorEmpty? voiceActivityDetectorEmpty;

  static SherpaOnnxVoiceActivityDetectorDetected? voiceActivityDetectorDetected;

  static SherpaOnnxVoiceActivityDetectorPop? voiceActivityDetectorPop;

  static SherpaOnnxVoiceActivityDetectorClear? voiceActivityDetectorClear;

  static SherpaOnnxVoiceActivityDetectorFront? voiceActivityDetectorFront;

  static SherpaOnnxDestroySpeechSegment? destroySpeechSegment;

  static SherpaOnnxVoiceActivityDetectorReset? voiceActivityDetectorReset;

  static SherpaOnnxCreateCircularBuffer? createCircularBuffer;

  static SherpaOnnxDestroyCircularBuffer? destroyCircularBuffer;

  static SherpaOnnxCircularBufferPush? circularBufferPush;

  static SherpaOnnxCircularBufferGet? circularBufferGet;

  static SherpaOnnxCircularBufferFree? circularBufferFree;

  static SherpaOnnxCircularBufferPop? circularBufferPop;

  static SherpaOnnxCircularBufferSize? circularBufferSize;

  static SherpaOnnxCircularBufferHead? circularBufferHead;

  static SherpaOnnxCircularBufferReset? circularBufferReset;

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

  static SherpaOnnxWriteWave? writeWave;

  static SherpaOnnxFreeWave? freeWave;

  static void init(DynamicLibrary dynamicLibrary) {
    createOfflineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<CreateOfflineRecognizerNative>>(
            'CreateOfflineRecognizer')
        .asFunction();

    destroyOfflineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOfflineRecognizerNative>>(
            'DestroyOfflineRecognizer')
        .asFunction();

    createOfflineStream ??= dynamicLibrary
        .lookup<NativeFunction<CreateOfflineStreamNative>>(
            'CreateOfflineStream')
        .asFunction();

    destroyOfflineStream ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOfflineStreamNative>>(
            'DestroyOfflineStream')
        .asFunction();

    acceptWaveformOffline ??= dynamicLibrary
        .lookup<NativeFunction<AcceptWaveformOfflineNative>>(
            'AcceptWaveformOffline')
        .asFunction();

    decodeOfflineStream ??= dynamicLibrary
        .lookup<NativeFunction<DecodeOfflineStreamNative>>(
            'DecodeOfflineStream')
        .asFunction();

    getOfflineStreamResultAsJson ??= dynamicLibrary
        .lookup<NativeFunction<GetOfflineStreamResultAsJsonNative>>(
            'GetOfflineStreamResultAsJson')
        .asFunction();

    destroyOfflineStreamResultJson ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOfflineStreamResultJsonNative>>(
            'DestroyOfflineStreamResultJson')
        .asFunction();

    createOnlineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<CreateOnlineRecognizerNative>>(
            'CreateOnlineRecognizer')
        .asFunction();

    destroyOnlineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOnlineRecognizerNative>>(
            'DestroyOnlineRecognizer')
        .asFunction();

    createOnlineStream ??= dynamicLibrary
        .lookup<NativeFunction<CreateOnlineStreamNative>>('CreateOnlineStream')
        .asFunction();

    createOnlineStreamWithHotwords ??= dynamicLibrary
        .lookup<NativeFunction<CreateOnlineStreamWithHotwordsNative>>(
            'CreateOnlineStreamWithHotwords')
        .asFunction();

    isOnlineStreamReady ??= dynamicLibrary
        .lookup<NativeFunction<IsOnlineStreamReadyNative>>(
            'IsOnlineStreamReady')
        .asFunction();

    decodeOnlineStream ??= dynamicLibrary
        .lookup<NativeFunction<DecodeOnlineStreamNative>>('DecodeOnlineStream')
        .asFunction();

    getOnlineStreamResultAsJson ??= dynamicLibrary
        .lookup<NativeFunction<GetOnlineStreamResultAsJsonNative>>(
            'GetOnlineStreamResultAsJson')
        .asFunction();

    reset ??= dynamicLibrary
        .lookup<NativeFunction<ResetNative>>('Reset')
        .asFunction();

    isEndpoint ??= dynamicLibrary
        .lookup<NativeFunction<IsEndpointNative>>('IsEndpoint')
        .asFunction();

    destroyOnlineStreamResultJson ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOnlineStreamResultJsonNative>>(
            'DestroyOnlineStreamResultJson')
        .asFunction();

    createVoiceActivityDetector ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateVoiceActivityDetectorNative>>(
            'SherpaOnnxCreateVoiceActivityDetector')
        .asFunction();

    destroyVoiceActivityDetector ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyVoiceActivityDetectorNative>>(
            'SherpaOnnxDestroyVoiceActivityDetector')
        .asFunction();

    voiceActivityDetectorAcceptWaveform ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxVoiceActivityDetectorAcceptWaveformNative>>(
            'SherpaOnnxVoiceActivityDetectorAcceptWaveform')
        .asFunction();

    voiceActivityDetectorEmpty ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorEmptyNative>>(
            'SherpaOnnxVoiceActivityDetectorEmpty')
        .asFunction();

    voiceActivityDetectorDetected ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorDetectedNative>>(
            'SherpaOnnxVoiceActivityDetectorDetected')
        .asFunction();

    voiceActivityDetectorPop ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorPopNative>>(
            'SherpaOnnxVoiceActivityDetectorPop')
        .asFunction();

    voiceActivityDetectorClear ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorClearNative>>(
            'SherpaOnnxVoiceActivityDetectorClear')
        .asFunction();

    voiceActivityDetectorFront ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorFrontNative>>(
            'SherpaOnnxVoiceActivityDetectorFront')
        .asFunction();

    destroySpeechSegment ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroySpeechSegmentNative>>(
            'SherpaOnnxDestroySpeechSegment')
        .asFunction();

    voiceActivityDetectorReset ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorResetNative>>(
            'SherpaOnnxVoiceActivityDetectorReset')
        .asFunction();

    createCircularBuffer ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateCircularBufferNative>>(
            'SherpaOnnxCreateCircularBuffer')
        .asFunction();

    destroyCircularBuffer ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyCircularBufferNative>>(
            'SherpaOnnxDestroyCircularBuffer')
        .asFunction();

    circularBufferPush ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferPushNative>>(
            'SherpaOnnxCircularBufferPush')
        .asFunction();

    circularBufferGet ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferGetNative>>(
            'SherpaOnnxCircularBufferGet')
        .asFunction();

    circularBufferFree ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferFreeNative>>(
            'SherpaOnnxCircularBufferFree')
        .asFunction();

    circularBufferPop ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferPopNative>>(
            'SherpaOnnxCircularBufferPop')
        .asFunction();

    circularBufferSize ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferSizeNative>>(
            'SherpaOnnxCircularBufferSize')
        .asFunction();

    circularBufferHead ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferHeadNative>>(
            'SherpaOnnxCircularBufferHead')
        .asFunction();

    circularBufferReset ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCircularBufferResetNative>>(
            'SherpaOnnxCircularBufferReset')
        .asFunction();

    createSpeakerEmbeddingExtractor ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxCreateSpeakerEmbeddingExtractorNative>>(
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

    writeWave ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxWriteWaveNative>>(
            'SherpaOnnxWriteWave')
        .asFunction();

    freeWave ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxFreeWaveNative>>('SherpaOnnxFreeWave')
        .asFunction();
  }
}
