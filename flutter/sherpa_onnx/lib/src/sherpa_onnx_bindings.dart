// Copyright (c)  2024  Xiaomi Corporation
import 'dart:ffi';
import 'package:ffi/ffi.dart';

final class SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineSpeechDenoiserModelConfig extends Struct {
  external SherpaOnnxOfflineSpeechDenoiserGtcrnModelConfig gtcrn;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxOfflineSpeechDenoiserConfig extends Struct {
  external SherpaOnnxOfflineSpeechDenoiserModelConfig model;
}

final class SherpaOnnxDenoisedAudio extends Struct {
  external Pointer<Float> samples;

  @Int32()
  external int n;

  @Int32()
  external int sampleRate;
}

final class SherpaOnnxSpeakerEmbeddingExtractorConfig extends Struct {
  external Pointer<Utf8> model;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxOfflineSpeakerDiarizationSegment extends Struct {
  @Float()
  external double start;

  @Float()
  external double end;

  @Int32()
  external int speaker;
}

final class SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig
    extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineSpeakerSegmentationModelConfig extends Struct {
  external SherpaOnnxOfflineSpeakerSegmentationPyannoteModelConfig pyannote;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxFastClusteringConfig extends Struct {
  @Int32()
  external int numClusters;

  @Float()
  external double threshold;
}

final class SherpaOnnxOfflineSpeakerDiarizationConfig extends Struct {
  external SherpaOnnxOfflineSpeakerSegmentationModelConfig segmentation;
  external SherpaOnnxSpeakerEmbeddingExtractorConfig embedding;
  external SherpaOnnxFastClusteringConfig clustering;

  @Float()
  external double minDurationOn;

  @Float()
  external double minDurationOff;
}

final class SherpaOnnxOfflinePunctuationModelConfig extends Struct {
  external Pointer<Utf8> ctTransformer;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxOfflinePunctuationConfig extends Struct {
  external SherpaOnnxOfflinePunctuationModelConfig model;
}

final class SherpaOnnxOnlinePunctuationModelConfig extends Struct {
  external Pointer<Utf8> cnnBiLstm;
  external Pointer<Utf8> bpeVocab;
  @Int32()
  external int numThreads;
  @Int32()
  external int debug;
  external Pointer<Utf8> provider;
}

final class SherpaOnnxOnlinePunctuationConfig extends Struct {
  external SherpaOnnxOnlinePunctuationModelConfig model;
}

final class SherpaOnnxOfflineZipformerAudioTaggingModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxAudioTaggingModelConfig extends Struct {
  external SherpaOnnxOfflineZipformerAudioTaggingModelConfig zipformer;
  external Pointer<Utf8> ced;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxAudioTaggingConfig extends Struct {
  external SherpaOnnxAudioTaggingModelConfig model;
  external Pointer<Utf8> labels;

  @Int32()
  external int topK;
}

final class SherpaOnnxAudioEvent extends Struct {
  external Pointer<Utf8> name;

  @Int32()
  external int index;

  @Float()
  external double prob;
}

final class SherpaOnnxOfflineTtsVitsModelConfig extends Struct {
  external Pointer<Utf8> model;
  external Pointer<Utf8> lexicon;
  external Pointer<Utf8> tokens;
  external Pointer<Utf8> dataDir;

  @Float()
  external double noiseScale;

  @Float()
  external double noiseScaleW;

  @Float()
  external double lengthScale;

  external Pointer<Utf8> dictDir;
}

final class SherpaOnnxOfflineTtsMatchaModelConfig extends Struct {
  external Pointer<Utf8> acousticModel;
  external Pointer<Utf8> vocoder;
  external Pointer<Utf8> lexicon;
  external Pointer<Utf8> tokens;
  external Pointer<Utf8> dataDir;

  @Float()
  external double noiseScale;

  @Float()
  external double lengthScale;

  external Pointer<Utf8> dictDir;
}

final class SherpaOnnxOfflineTtsKokoroModelConfig extends Struct {
  external Pointer<Utf8> model;
  external Pointer<Utf8> voices;
  external Pointer<Utf8> tokens;
  external Pointer<Utf8> dataDir;

  @Float()
  external double lengthScale;
  external Pointer<Utf8> dictDir;
  external Pointer<Utf8> lexicon;
  external Pointer<Utf8> lang;
}

final class SherpaOnnxOfflineTtsKittenModelConfig extends Struct {
  external Pointer<Utf8> model;
  external Pointer<Utf8> voices;
  external Pointer<Utf8> tokens;
  external Pointer<Utf8> dataDir;

  @Float()
  external double lengthScale;
}

final class SherpaOnnxOfflineTtsZipVoiceModelConfig extends Struct {
  external Pointer<Utf8> tokens;
  external Pointer<Utf8> textModel;
  external Pointer<Utf8> flowMatchingModel;
  external Pointer<Utf8> vocoder;
  external Pointer<Utf8> dataDir;
  external Pointer<Utf8> pinyinDict;

  @Float()
  external double featScale;

  @Float()
  external double tShift;

  @Float()
  external double targetRms;

  @Float()
  external double guidanceScale;
}

final class SherpaOnnxOfflineTtsModelConfig extends Struct {
  external SherpaOnnxOfflineTtsVitsModelConfig vits;
  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
  external SherpaOnnxOfflineTtsMatchaModelConfig matcha;
  external SherpaOnnxOfflineTtsKokoroModelConfig kokoro;
  external SherpaOnnxOfflineTtsKittenModelConfig kitten;
  external SherpaOnnxOfflineTtsZipVoiceModelConfig zipvoice;
}

final class SherpaOnnxOfflineTtsConfig extends Struct {
  external SherpaOnnxOfflineTtsModelConfig model;
  external Pointer<Utf8> ruleFsts;

  @Int32()
  external int maxNumSenetences;

  external Pointer<Utf8> ruleFars;

  @Float()
  external double silenceScale;
}

final class SherpaOnnxGeneratedAudio extends Struct {
  external Pointer<Float> samples;

  @Int32()
  external int n;

  @Int32()
  external int sampleRate;
}

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

final class SherpaOnnxOfflineDolphinModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineZipformerCtcModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineWenetCtcModelConfig extends Struct {
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

final class SherpaOnnxOfflineCanaryModelConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;
  external Pointer<Utf8> srcLang;
  external Pointer<Utf8> tgtLang;

  @Int32()
  external int usePnc;
}

final class SherpaOnnxOfflineMoonshineModelConfig extends Struct {
  external Pointer<Utf8> preprocessor;
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> uncachedDecoder;
  external Pointer<Utf8> cachedDecoder;
}

final class SherpaOnnxOfflineFireRedAsrModelConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;
}

final class SherpaOnnxOfflineTdnnModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOfflineSenseVoiceModelConfig extends Struct {
  external Pointer<Utf8> model;
  external Pointer<Utf8> language;

  @Int32()
  external int useInverseTextNormalization;
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
  external Pointer<Utf8> modelingUnit;
  external Pointer<Utf8> bpeVocab;
  external Pointer<Utf8> telespeechCtc;

  external SherpaOnnxOfflineSenseVoiceModelConfig senseVoice;
  external SherpaOnnxOfflineMoonshineModelConfig moonshine;
  external SherpaOnnxOfflineFireRedAsrModelConfig fireRedAsr;
  external SherpaOnnxOfflineDolphinModelConfig dolphin;
  external SherpaOnnxOfflineZipformerCtcModelConfig zipformerCtc;
  external SherpaOnnxOfflineCanaryModelConfig canary;
  external SherpaOnnxOfflineWenetCtcModelConfig wenetCtc;
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

  external Pointer<Utf8> ruleFsts;
  external Pointer<Utf8> ruleFars;

  @Float()
  external double blankPenalty;
  external SherpaOnnxHomophoneReplacerConfig hr;
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

final class SherpaOnnxOnlineNemoCtcModelConfig extends Struct {
  external Pointer<Utf8> model;
}

final class SherpaOnnxOnlineToneCtcModelConfig extends Struct {
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

  external Pointer<Utf8> modelingUnit;

  external Pointer<Utf8> bpeVocab;

  external Pointer<Utf8> tokensBuf;

  @Int32()
  external int tokensBufSize;

  external SherpaOnnxOnlineNemoCtcModelConfig nemoCtc;

  external SherpaOnnxOnlineToneCtcModelConfig toneCtc;
}

final class SherpaOnnxOnlineCtcFstDecoderConfig extends Struct {
  external Pointer<Utf8> graph;

  @Int32()
  external int maxActive;
}

final class SherpaOnnxHomophoneReplacerConfig extends Struct {
  external Pointer<Utf8> dictDir;
  external Pointer<Utf8> lexicon;
  external Pointer<Utf8> ruleFsts;
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

  external Pointer<Utf8> ruleFsts;
  external Pointer<Utf8> ruleFars;

  @Float()
  external double blankPenalty;

  external Pointer<Utf8> hotwordsBuf;

  @Int32()
  external int hotwordsBufSize;
  external SherpaOnnxHomophoneReplacerConfig hr;
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

  @Float()
  external double maxSpeechDuration;
}

final class SherpaOnnxTenVadModelConfig extends Struct {
  external Pointer<Utf8> model;

  @Float()
  external double threshold;

  @Float()
  external double minSilenceDuration;

  @Float()
  external double minSpeechDuration;

  @Int32()
  external int windowSize;

  @Float()
  external double maxSpeechDuration;
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

  external SherpaOnnxTenVadModelConfig tenVad;
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

final class SherpaOnnxKeywordSpotterConfig extends Struct {
  external SherpaOnnxFeatureConfig feat;

  external SherpaOnnxOnlineModelConfig model;

  @Int32()
  external int maxActivePaths;

  @Int32()
  external int numTrailingBlanks;

  @Float()
  external double keywordsScore;

  @Float()
  external double keywordsThreshold;

  external Pointer<Utf8> keywordsFile;

  external Pointer<Utf8> keywordsBuf;

  @Int32()
  external int keywordsBufSize;
}

final class SherpaOnnxOfflinePunctuation extends Opaque {}

final class SherpaOnnxOnlinePunctuation extends Opaque {}

final class SherpaOnnxAudioTagging extends Opaque {}

final class SherpaOnnxKeywordSpotter extends Opaque {}

final class SherpaOnnxOfflineTts extends Opaque {}

final class SherpaOnnxCircularBuffer extends Opaque {}

final class SherpaOnnxVoiceActivityDetector extends Opaque {}

final class SherpaOnnxOnlineStream extends Opaque {}

final class SherpaOnnxOnlineRecognizer extends Opaque {}

final class SherpaOnnxOfflineRecognizer extends Opaque {}

final class SherpaOnnxOfflineStream extends Opaque {}

final class SherpaOnnxSpeakerEmbeddingExtractor extends Opaque {}

final class SherpaOnnxSpeakerEmbeddingManager extends Opaque {}

final class SherpaOnnxOfflineSpeakerDiarization extends Opaque {}

final class SherpaOnnxOfflineSpeakerDiarizationResult extends Opaque {}

final class SherpaOnnxSpokenLanguageIdentificationWhisperConfig extends Struct {
  external Pointer<Utf8> encoder;
  external Pointer<Utf8> decoder;

  @Int32()
  external int tailPaddings;
}

final class SherpaOnnxSpokenLanguageIdentificationConfig extends Struct {
  external SherpaOnnxSpokenLanguageIdentificationWhisperConfig whisper;

  @Int32()
  external int numThreads;

  @Int32()
  external int debug;

  external Pointer<Utf8> provider;
}

final class SherpaOnnxSpokenLanguageIdentificationResult extends Struct {
  external Pointer<Utf8> lang;
}

final class SherpaOnnxSpokenLanguageIdentification extends Opaque {}

final class SherpaOnnxOfflineSpeechDenoiser extends Opaque {}

typedef SherpaOnnxCreateOfflineSpeechDenoiserNative
    = Pointer<SherpaOnnxOfflineSpeechDenoiser> Function(
        Pointer<SherpaOnnxOfflineSpeechDenoiserConfig>);

typedef SherpaOnnxCreateOfflineSpeechDenoiser
    = SherpaOnnxCreateOfflineSpeechDenoiserNative;

typedef SherpaOnnxDestroyOfflineSpeechDenoiserNative = Void Function(
    Pointer<SherpaOnnxOfflineSpeechDenoiser>);

typedef SherpaOnnxDestroyOfflineSpeechDenoiser = void Function(
    Pointer<SherpaOnnxOfflineSpeechDenoiser>);

typedef SherpaOnnxOfflineSpeechDenoiserGetSampleRateNative = Int32 Function(
    Pointer<SherpaOnnxOfflineSpeechDenoiser>);

typedef SherpaOnnxOfflineSpeechDenoiserGetSampleRate = int Function(
    Pointer<SherpaOnnxOfflineSpeechDenoiser>);

typedef SherpaOnnxOfflineSpeechDenoiserRunNative
    = Pointer<SherpaOnnxDenoisedAudio> Function(
        Pointer<SherpaOnnxOfflineSpeechDenoiser>, Pointer<Float>, Int32, Int32);

typedef SherpaOnnxOfflineSpeechDenoiserRun
    = Pointer<SherpaOnnxDenoisedAudio> Function(
        Pointer<SherpaOnnxOfflineSpeechDenoiser>, Pointer<Float>, int, int);

typedef SherpaOnnxDestroyDenoisedAudioNative = Void Function(
    Pointer<SherpaOnnxDenoisedAudio>);

typedef SherpaOnnxDestroyDenoisedAudio = void Function(
    Pointer<SherpaOnnxDenoisedAudio>);

typedef SherpaOnnxCreateSpokenLanguageIdentificationNative
    = Pointer<SherpaOnnxSpokenLanguageIdentification> Function(
        Pointer<SherpaOnnxSpokenLanguageIdentificationConfig>);

typedef SherpaOnnxCreateSpokenLanguageIdentification
    = SherpaOnnxCreateSpokenLanguageIdentificationNative;

typedef SherpaOnnxDestroySpokenLanguageIdentificationNative = Void Function(
    Pointer<SherpaOnnxSpokenLanguageIdentification>);

typedef SherpaOnnxDestroySpokenLanguageIdentification = void Function(
    Pointer<SherpaOnnxSpokenLanguageIdentification>);

typedef SherpaOnnxSpokenLanguageIdentificationCreateOfflineStreamNative
    = Pointer<SherpaOnnxOfflineStream> Function(
        Pointer<SherpaOnnxSpokenLanguageIdentification>);

typedef SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream
    = SherpaOnnxSpokenLanguageIdentificationCreateOfflineStreamNative;

typedef SherpaOnnxSpokenLanguageIdentificationComputeNative
    = Pointer<SherpaOnnxSpokenLanguageIdentificationResult> Function(
        Pointer<SherpaOnnxSpokenLanguageIdentification>,
        Pointer<SherpaOnnxOfflineStream>);

typedef SherpaOnnxSpokenLanguageIdentificationCompute
    = SherpaOnnxSpokenLanguageIdentificationComputeNative;

typedef SherpaOnnxDestroySpokenLanguageIdentificationResultNative = Void
    Function(Pointer<SherpaOnnxSpokenLanguageIdentificationResult>);

typedef SherpaOnnxDestroySpokenLanguageIdentificationResult = void Function(
    Pointer<SherpaOnnxSpokenLanguageIdentificationResult>);

typedef SherpaOnnxCreateOfflineSpeakerDiarizationNative
    = Pointer<SherpaOnnxOfflineSpeakerDiarization> Function(
        Pointer<SherpaOnnxOfflineSpeakerDiarizationConfig>);

typedef SherpaOnnxCreateOfflineSpeakerDiarization
    = SherpaOnnxCreateOfflineSpeakerDiarizationNative;

typedef SherpaOnnxDestroyOfflineSpeakerDiarizationNative = Void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarization>);

typedef SherpaOnnxDestroyOfflineSpeakerDiarization = void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarization>);

typedef SherpaOnnxCreateOfflinePunctuationNative
    = Pointer<SherpaOnnxOfflinePunctuation> Function(
        Pointer<SherpaOnnxOfflinePunctuationConfig>);

typedef SherpaOnnxCreateOnlinePunctuationNative
    = Pointer<SherpaOnnxOnlinePunctuation> Function(
        Pointer<SherpaOnnxOnlinePunctuationConfig>);

typedef SherpaOnnxOfflineSpeakerDiarizationGetSampleRateNative = Int32 Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarization>);

typedef SherpaOnnxOfflineSpeakerDiarizationGetSampleRate = int Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarization>);

typedef SherpaOnnxOfflineSpeakerDiarizationSetConfigNative = Void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarization>,
    Pointer<SherpaOnnxOfflineSpeakerDiarizationConfig>);

typedef SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakersNative = Int32
    Function(Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers = int Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegmentsNative = Int32
    Function(Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments = int Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTimeNative
    = Pointer<SherpaOnnxOfflineSpeakerDiarizationSegment> Function(
        Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime
    = SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTimeNative;

typedef SherpaOnnxOfflineSpeakerDiarizationDestroySegmentNative = Void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarizationSegment>);

typedef SherpaOnnxOfflineSpeakerDiarizationDestroySegment = void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarizationSegment>);

typedef SherpaOnnxOfflineSpeakerDiarizationProcessNative
    = Pointer<SherpaOnnxOfflineSpeakerDiarizationResult> Function(
        Pointer<SherpaOnnxOfflineSpeakerDiarization>, Pointer<Float>, Int32);

typedef SherpaOnnxOfflineSpeakerDiarizationProcess
    = Pointer<SherpaOnnxOfflineSpeakerDiarizationResult> Function(
        Pointer<SherpaOnnxOfflineSpeakerDiarization>, Pointer<Float>, int);

typedef SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArgNative = Int32
    Function(Int32, Int32);

typedef SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArgNative
    = Pointer<SherpaOnnxOfflineSpeakerDiarizationResult> Function(
        Pointer<SherpaOnnxOfflineSpeakerDiarization>,
        Pointer<Float>,
        Int32,
        Pointer<
            NativeFunction<
                SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArgNative>>);

typedef SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg
    = Pointer<SherpaOnnxOfflineSpeakerDiarizationResult> Function(
        Pointer<SherpaOnnxOfflineSpeakerDiarization>,
        Pointer<Float>,
        int,
        Pointer<
            NativeFunction<
                SherpaOnnxOfflineSpeakerDiarizationProgressCallbackNoArgNative>>);

typedef SherpaOnnxOfflineSpeakerDiarizationDestroyResultNative = Void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationDestroyResult = void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarizationResult>);

typedef SherpaOnnxOfflineSpeakerDiarizationSetConfig = void Function(
    Pointer<SherpaOnnxOfflineSpeakerDiarization>,
    Pointer<SherpaOnnxOfflineSpeakerDiarizationConfig>);

typedef SherpaOnnxCreateOfflinePunctuation
    = SherpaOnnxCreateOfflinePunctuationNative;

typedef SherpaOnnxDestroyOfflinePunctuationNative = Void Function(
    Pointer<SherpaOnnxOfflinePunctuation>);

typedef SherpaOnnxDestroyOfflinePunctuation = void Function(
    Pointer<SherpaOnnxOfflinePunctuation>);

typedef SherpaOfflinePunctuationAddPunctNative = Pointer<Utf8> Function(
    Pointer<SherpaOnnxOfflinePunctuation>, Pointer<Utf8>);

typedef SherpaOfflinePunctuationAddPunct
    = SherpaOfflinePunctuationAddPunctNative;

typedef SherpaOfflinePunctuationFreeTextNative = Void Function(Pointer<Utf8>);

typedef SherpaOfflinePunctuationFreeText = void Function(Pointer<Utf8>);

typedef SherpaOnnxCreateOnlinePunctuation
    = SherpaOnnxCreateOnlinePunctuationNative;

typedef SherpaOnnxDestroyOnlinePunctuationNative = Void Function(
    Pointer<SherpaOnnxOnlinePunctuation>);

typedef SherpaOnnxDestroyOnlinePunctuation = void Function(
    Pointer<SherpaOnnxOnlinePunctuation>);

typedef SherpaOnnxOnlinePunctuationAddPunctNative = Pointer<Utf8> Function(
    Pointer<SherpaOnnxOnlinePunctuation>, Pointer<Utf8>);

typedef SherpaOnnxOnlinePunctuationAddPunct
    = SherpaOnnxOnlinePunctuationAddPunctNative;

typedef SherpaOnnxOnlinePunctuationFreeTextNative = Void Function(
    Pointer<Utf8>);

typedef SherpaOnnxOnlinePunctuationFreeText = void Function(Pointer<Utf8>);

typedef SherpaOnnxCreateAudioTaggingNative = Pointer<SherpaOnnxAudioTagging>
    Function(Pointer<SherpaOnnxAudioTaggingConfig>);

typedef SherpaOnnxCreateAudioTagging = SherpaOnnxCreateAudioTaggingNative;

typedef SherpaOnnxDestroyAudioTaggingNative = Void Function(
    Pointer<SherpaOnnxAudioTagging>);

typedef SherpaOnnxDestroyAudioTagging = void Function(
    Pointer<SherpaOnnxAudioTagging>);

typedef SherpaOnnxAudioTaggingCreateOfflineStreamNative
    = Pointer<SherpaOnnxOfflineStream> Function(
        Pointer<SherpaOnnxAudioTagging>);

typedef SherpaOnnxAudioTaggingCreateOfflineStream
    = SherpaOnnxAudioTaggingCreateOfflineStreamNative;

typedef SherpaOnnxAudioTaggingComputeNative
    = Pointer<Pointer<SherpaOnnxAudioEvent>> Function(
        Pointer<SherpaOnnxAudioTagging>,
        Pointer<SherpaOnnxOfflineStream>,
        Int32);

typedef SherpaOnnxAudioTaggingCompute
    = Pointer<Pointer<SherpaOnnxAudioEvent>> Function(
        Pointer<SherpaOnnxAudioTagging>, Pointer<SherpaOnnxOfflineStream>, int);

typedef SherpaOnnxAudioTaggingFreeResultsNative = Void Function(
    Pointer<Pointer<SherpaOnnxAudioEvent>>);

typedef SherpaOnnxAudioTaggingFreeResults = void Function(
    Pointer<Pointer<SherpaOnnxAudioEvent>>);

typedef CreateKeywordSpotterNative = Pointer<SherpaOnnxKeywordSpotter> Function(
    Pointer<SherpaOnnxKeywordSpotterConfig>);

typedef CreateKeywordSpotter = CreateKeywordSpotterNative;

typedef DestroyKeywordSpotterNative = Void Function(
    Pointer<SherpaOnnxKeywordSpotter>);

typedef DestroyKeywordSpotter = void Function(
    Pointer<SherpaOnnxKeywordSpotter>);

typedef CreateKeywordStreamNative = Pointer<SherpaOnnxOnlineStream> Function(
    Pointer<SherpaOnnxKeywordSpotter>);

typedef CreateKeywordStream = CreateKeywordStreamNative;

typedef CreateKeywordStreamWithKeywordsNative = Pointer<SherpaOnnxOnlineStream>
    Function(Pointer<SherpaOnnxKeywordSpotter>, Pointer<Utf8>);

typedef CreateKeywordStreamWithKeywords = CreateKeywordStreamWithKeywordsNative;

typedef IsKeywordStreamReadyNative = Int32 Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef IsKeywordStreamReady = int Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef DecodeKeywordStreamNative = Void Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef DecodeKeywordStream = void Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef ResetKeywordStreamNative = Void Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef ResetKeywordStream = void Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef GetKeywordResultAsJsonNative = Pointer<Utf8> Function(
    Pointer<SherpaOnnxKeywordSpotter>, Pointer<SherpaOnnxOnlineStream>);

typedef GetKeywordResultAsJson = GetKeywordResultAsJsonNative;

typedef FreeKeywordResultJsonNative = Void Function(Pointer<Utf8>);

typedef FreeKeywordResultJson = void Function(Pointer<Utf8>);

typedef SherpaOnnxCreateOfflineTtsNative = Pointer<SherpaOnnxOfflineTts>
    Function(Pointer<SherpaOnnxOfflineTtsConfig>);

typedef SherpaOnnxCreateOfflineTts = SherpaOnnxCreateOfflineTtsNative;

typedef SherpaOnnxDestroyOfflineTtsNative = Void Function(
    Pointer<SherpaOnnxOfflineTts>);

typedef SherpaOnnxDestroyOfflineTts = void Function(
    Pointer<SherpaOnnxOfflineTts>);

typedef SherpaOnnxOfflineTtsSampleRateNative = Int32 Function(
    Pointer<SherpaOnnxOfflineTts>);

typedef SherpaOnnxOfflineTtsSampleRate = int Function(
    Pointer<SherpaOnnxOfflineTts>);

typedef SherpaOnnxOfflineTtsNumSpeakersNative = Int32 Function(
    Pointer<SherpaOnnxOfflineTts>);

typedef SherpaOnnxOfflineTtsNumSpeakers = int Function(
    Pointer<SherpaOnnxOfflineTts>);

typedef SherpaOnnxOfflineTtsGenerateNative = Pointer<SherpaOnnxGeneratedAudio>
    Function(Pointer<SherpaOnnxOfflineTts>, Pointer<Utf8>, Int32, Float);

typedef SherpaOnnxOfflineTtsGenerate = Pointer<SherpaOnnxGeneratedAudio>
    Function(Pointer<SherpaOnnxOfflineTts>, Pointer<Utf8>, int, double);

typedef SherpaOnnxDestroyOfflineTtsGeneratedAudioNative = Void Function(
    Pointer<SherpaOnnxGeneratedAudio>);

typedef SherpaOnnxDestroyOfflineTtsGeneratedAudio = void Function(
    Pointer<SherpaOnnxGeneratedAudio>);

typedef SherpaOnnxGeneratedAudioCallbackNative = Int Function(
    Pointer<Float>, Int32);

typedef SherpaOnnxOfflineTtsGenerateWithCallbackNative
    = Pointer<SherpaOnnxGeneratedAudio> Function(
        Pointer<SherpaOnnxOfflineTts>,
        Pointer<Utf8>,
        Int32,
        Float,
        Pointer<NativeFunction<SherpaOnnxGeneratedAudioCallbackNative>>);

typedef SherpaOnnxOfflineTtsGenerateWithCallback
    = Pointer<SherpaOnnxGeneratedAudio> Function(
        Pointer<SherpaOnnxOfflineTts>,
        Pointer<Utf8>,
        int,
        double,
        Pointer<NativeFunction<SherpaOnnxGeneratedAudioCallbackNative>>);

typedef CreateOfflineRecognizerNative = Pointer<SherpaOnnxOfflineRecognizer>
    Function(Pointer<SherpaOnnxOfflineRecognizerConfig>);

typedef CreateOfflineRecognizer = CreateOfflineRecognizerNative;

typedef OfflineRecognizerSetConfigNative = Void Function(
    Pointer<SherpaOnnxOfflineRecognizer>,
    Pointer<SherpaOnnxOfflineRecognizerConfig>);

typedef OfflineRecognizerSetConfig = void Function(
    Pointer<SherpaOnnxOfflineRecognizer>,
    Pointer<SherpaOnnxOfflineRecognizerConfig>);

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

typedef SherpaOnnxCreateOnlineRecognizerNative
    = Pointer<SherpaOnnxOnlineRecognizer> Function(
        Pointer<SherpaOnnxOnlineRecognizerConfig>);

typedef SherpaOnnxCreateOnlineRecognizer
    = SherpaOnnxCreateOnlineRecognizerNative;

typedef SherpaOnnxDestroyOnlineRecognizerNative = Void Function(
    Pointer<SherpaOnnxOnlineRecognizer>);

typedef SherpaOnnxDestroyOnlineRecognizer = void Function(
    Pointer<SherpaOnnxOnlineRecognizer>);

typedef SherpaOnnxCreateOnlineStreamNative = Pointer<SherpaOnnxOnlineStream>
    Function(Pointer<SherpaOnnxOnlineRecognizer>);

typedef SherpaOnnxCreateOnlineStream = SherpaOnnxCreateOnlineStreamNative;

typedef SherpaOnnxCreateOnlineStreamWithHotwordsNative
    = Pointer<SherpaOnnxOnlineStream> Function(
        Pointer<SherpaOnnxOnlineRecognizer>, Pointer<Utf8>);

typedef SherpaOnnxCreateOnlineStreamWithHotwords
    = SherpaOnnxCreateOnlineStreamWithHotwordsNative;

typedef IsOnlineStreamReadyNative = Int32 Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef IsOnlineStreamReady = int Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxDecodeOnlineStreamNative = Void Function(
    Pointer<SherpaOnnxOnlineRecognizer>, Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxDecodeOnlineStream = void Function(
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

typedef SherpaOnnxVoiceActivityDetectorFlushNative = Void Function(
    Pointer<SherpaOnnxVoiceActivityDetector>);

typedef SherpaOnnxVoiceActivityDetectorFlush = void Function(
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

typedef SherpaOnnxDestroyOnlineStreamNative = Void Function(
    Pointer<SherpaOnnxOnlineStream>);

typedef SherpaOnnxDestroyOnlineStream = void Function(
    Pointer<SherpaOnnxOnlineStream>);

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

typedef SherpaOnnxGetVersionStr = Pointer<Utf8> Function();
typedef SherpaOnnxGetVersionStrNative = SherpaOnnxGetVersionStr;

typedef SherpaOnnxGetGitSha1Native = Pointer<Utf8> Function();
typedef SherpaOnnxGetGitSha1 = SherpaOnnxGetGitSha1Native;

typedef SherpaOnnxGetGitDateNative = Pointer<Utf8> Function();
typedef SherpaOnnxGetGitDate = SherpaOnnxGetGitDateNative;

class SherpaOnnxBindings {
  static SherpaOnnxCreateOfflineSpeechDenoiser?
      sherpaOnnxCreateOfflineSpeechDenoiser;

  static SherpaOnnxDestroyOfflineSpeechDenoiser?
      sherpaOnnxDestroyOfflineSpeechDenoiser;

  static SherpaOnnxOfflineSpeechDenoiserGetSampleRate?
      sherpaOnnxOfflineSpeechDenoiserGetSampleRate;
  static SherpaOnnxOfflineSpeechDenoiserRun? sherpaOnnxOfflineSpeechDenoiserRun;
  static SherpaOnnxDestroyDenoisedAudio? sherpaOnnxDestroyDenoisedAudio;

  static SherpaOnnxCreateSpokenLanguageIdentification?
      sherpaOnnxCreateSpokenLanguageIdentification;
  static SherpaOnnxDestroySpokenLanguageIdentification?
      sherpaOnnxDestroySpokenLanguageIdentification;
  static SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream?
      sherpaOnnxSpokenLanguageIdentificationCreateOfflineStream;
  static SherpaOnnxSpokenLanguageIdentificationCompute?
      sherpaOnnxSpokenLanguageIdentificationCompute;
  static SherpaOnnxDestroySpokenLanguageIdentificationResult?
      sherpaOnnxDestroySpokenLanguageIdentificationResult;

  static SherpaOnnxCreateOfflineSpeakerDiarization?
      sherpaOnnxCreateOfflineSpeakerDiarization;
  static SherpaOnnxDestroyOfflineSpeakerDiarization?
      sherpaOnnxDestroyOfflineSpeakerDiarization;
  static SherpaOnnxOfflineSpeakerDiarizationGetSampleRate?
      sherpaOnnxOfflineSpeakerDiarizationGetSampleRate;
  static SherpaOnnxOfflineSpeakerDiarizationSetConfig?
      sherpaOnnxOfflineSpeakerDiarizationSetConfig;
  static SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers?
      sherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers;
  static SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments?
      sherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments;
  static SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime?
      sherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime;
  static SherpaOnnxOfflineSpeakerDiarizationDestroySegment?
      sherpaOnnxOfflineSpeakerDiarizationDestroySegment;
  static SherpaOnnxOfflineSpeakerDiarizationProcess?
      sherpaOnnxOfflineSpeakerDiarizationProcess;
  static SherpaOnnxOfflineSpeakerDiarizationDestroyResult?
      sherpaOnnxOfflineSpeakerDiarizationDestroyResult;
  static SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg?
      sherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg;

  static SherpaOnnxCreateOfflinePunctuation? sherpaOnnxCreateOfflinePunctuation;
  static SherpaOnnxDestroyOfflinePunctuation?
      sherpaOnnxDestroyOfflinePunctuation;
  static SherpaOfflinePunctuationAddPunct? sherpaOfflinePunctuationAddPunct;
  static SherpaOfflinePunctuationFreeText? sherpaOfflinePunctuationFreeText;

  static SherpaOnnxCreateOnlinePunctuation? sherpaOnnxCreateOnlinePunctuation;
  static SherpaOnnxDestroyOnlinePunctuation? sherpaOnnxDestroyOnlinePunctuation;
  static SherpaOnnxOnlinePunctuationAddPunct?
      sherpaOnnxOnlinePunctuationAddPunct;
  static SherpaOnnxOnlinePunctuationFreeText?
      sherpaOnnxOnlinePunctuationFreeText;

  static SherpaOnnxCreateAudioTagging? sherpaOnnxCreateAudioTagging;
  static SherpaOnnxDestroyAudioTagging? sherpaOnnxDestroyAudioTagging;
  static SherpaOnnxAudioTaggingCreateOfflineStream?
      sherpaOnnxAudioTaggingCreateOfflineStream;
  static SherpaOnnxAudioTaggingCompute? sherpaOnnxAudioTaggingCompute;
  static SherpaOnnxAudioTaggingFreeResults? sherpaOnnxAudioTaggingFreeResults;

  static CreateKeywordSpotter? createKeywordSpotter;
  static DestroyKeywordSpotter? destroyKeywordSpotter;
  static CreateKeywordStream? createKeywordStream;
  static CreateKeywordStreamWithKeywords? createKeywordStreamWithKeywords;
  static IsKeywordStreamReady? isKeywordStreamReady;
  static DecodeKeywordStream? decodeKeywordStream;
  static ResetKeywordStream? resetKeywordStream;
  static GetKeywordResultAsJson? getKeywordResultAsJson;
  static FreeKeywordResultJson? freeKeywordResultJson;

  static SherpaOnnxCreateOfflineTts? createOfflineTts;
  static SherpaOnnxDestroyOfflineTts? destroyOfflineTts;
  static SherpaOnnxOfflineTtsSampleRate? offlineTtsSampleRate;
  static SherpaOnnxOfflineTtsNumSpeakers? offlineTtsNumSpeakers;
  static SherpaOnnxOfflineTtsGenerate? offlineTtsGenerate;
  static SherpaOnnxDestroyOfflineTtsGeneratedAudio?
      destroyOfflineTtsGeneratedAudio;
  static SherpaOnnxOfflineTtsGenerateWithCallback?
      offlineTtsGenerateWithCallback;

  static CreateOfflineRecognizer? createOfflineRecognizer;
  static DestroyOfflineRecognizer? destroyOfflineRecognizer;
  static OfflineRecognizerSetConfig? offlineRecognizerSetConfig;
  static CreateOfflineStream? createOfflineStream;
  static DestroyOfflineStream? destroyOfflineStream;
  static AcceptWaveformOffline? acceptWaveformOffline;
  static DecodeOfflineStream? decodeOfflineStream;
  static GetOfflineStreamResultAsJson? getOfflineStreamResultAsJson;
  static DestroyOfflineStreamResultJson? destroyOfflineStreamResultJson;

  static SherpaOnnxCreateOnlineRecognizer? createOnlineRecognizer;

  static SherpaOnnxDestroyOnlineRecognizer? destroyOnlineRecognizer;

  static SherpaOnnxCreateOnlineStream? createOnlineStream;

  static SherpaOnnxCreateOnlineStreamWithHotwords?
      createOnlineStreamWithHotwords;

  static IsOnlineStreamReady? isOnlineStreamReady;

  static SherpaOnnxDecodeOnlineStream? decodeOnlineStream;

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

  static SherpaOnnxVoiceActivityDetectorFlush? voiceActivityDetectorFlush;

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

  static SherpaOnnxDestroyOnlineStream? destroyOnlineStream;

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

  static SherpaOnnxGetVersionStr? getVersionStr;
  static SherpaOnnxGetGitSha1? getGitSha1;
  static SherpaOnnxGetGitDate? getGitDate;

  static void init(DynamicLibrary dynamicLibrary) {
    sherpaOnnxCreateOfflineSpeechDenoiser ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOfflineSpeechDenoiserNative>>(
            'SherpaOnnxCreateOfflineSpeechDenoiser')
        .asFunction();

    sherpaOnnxDestroyOfflineSpeechDenoiser ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyOfflineSpeechDenoiserNative>>(
            'SherpaOnnxDestroyOfflineSpeechDenoiser')
        .asFunction();

    sherpaOnnxOfflineSpeechDenoiserGetSampleRate ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeechDenoiserGetSampleRateNative>>(
            'SherpaOnnxOfflineSpeechDenoiserGetSampleRate')
        .asFunction();

    sherpaOnnxOfflineSpeechDenoiserRun ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOfflineSpeechDenoiserRunNative>>(
            'SherpaOnnxOfflineSpeechDenoiserRun')
        .asFunction();

    sherpaOnnxDestroyDenoisedAudio ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyDenoisedAudioNative>>(
            'SherpaOnnxDestroyDenoisedAudio')
        .asFunction();

    sherpaOnnxCreateSpokenLanguageIdentification ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxCreateSpokenLanguageIdentificationNative>>(
            'SherpaOnnxCreateSpokenLanguageIdentification')
        .asFunction();

    sherpaOnnxDestroySpokenLanguageIdentification ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxDestroySpokenLanguageIdentificationNative>>(
            'SherpaOnnxDestroySpokenLanguageIdentification')
        .asFunction();

    sherpaOnnxSpokenLanguageIdentificationCreateOfflineStream ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpokenLanguageIdentificationCreateOfflineStreamNative>>(
            'SherpaOnnxSpokenLanguageIdentificationCreateOfflineStream')
        .asFunction();

    sherpaOnnxSpokenLanguageIdentificationCompute ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxSpokenLanguageIdentificationComputeNative>>(
            'SherpaOnnxSpokenLanguageIdentificationCompute')
        .asFunction();

    sherpaOnnxDestroySpokenLanguageIdentificationResult ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxDestroySpokenLanguageIdentificationResultNative>>(
            'SherpaOnnxDestroySpokenLanguageIdentificationResult')
        .asFunction();

    sherpaOnnxCreateOfflineSpeakerDiarization ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxCreateOfflineSpeakerDiarizationNative>>(
            'SherpaOnnxCreateOfflineSpeakerDiarization')
        .asFunction();

    sherpaOnnxDestroyOfflineSpeakerDiarization ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxDestroyOfflineSpeakerDiarizationNative>>(
            'SherpaOnnxDestroyOfflineSpeakerDiarization')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationGetSampleRate ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationGetSampleRateNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationGetSampleRate')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationSetConfig ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationSetConfigNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationSetConfig')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakersNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationResultGetNumSpeakers')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegmentsNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationResultGetNumSegments')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTimeNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationResultSortByStartTime')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationDestroySegment ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationDestroySegmentNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationDestroySegment')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationProcess ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationProcessNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationProcess')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArgNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationProcessWithCallbackNoArg')
        .asFunction();

    sherpaOnnxOfflineSpeakerDiarizationDestroyResult ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxOfflineSpeakerDiarizationDestroyResultNative>>(
            'SherpaOnnxOfflineSpeakerDiarizationDestroyResult')
        .asFunction();

    sherpaOnnxCreateOfflinePunctuation ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOfflinePunctuationNative>>(
            'SherpaOnnxCreateOfflinePunctuation')
        .asFunction();

    sherpaOnnxDestroyOfflinePunctuation ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyOfflinePunctuationNative>>(
            'SherpaOnnxDestroyOfflinePunctuation')
        .asFunction();

    sherpaOfflinePunctuationAddPunct ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOfflinePunctuationAddPunctNative>>(
            'SherpaOfflinePunctuationAddPunct')
        .asFunction();

    sherpaOfflinePunctuationFreeText ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOfflinePunctuationFreeTextNative>>(
            'SherpaOfflinePunctuationFreeText')
        .asFunction();

    sherpaOnnxCreateOnlinePunctuation ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOnlinePunctuationNative>>(
            'SherpaOnnxCreateOnlinePunctuation')
        .asFunction();

    sherpaOnnxDestroyOnlinePunctuation ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyOnlinePunctuationNative>>(
            'SherpaOnnxDestroyOnlinePunctuation')
        .asFunction();

    sherpaOnnxOnlinePunctuationAddPunct ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOnlinePunctuationAddPunctNative>>(
            'SherpaOnnxOnlinePunctuationAddPunct')
        .asFunction();

    sherpaOnnxOnlinePunctuationFreeText ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOnlinePunctuationFreeTextNative>>(
            'SherpaOnnxOnlinePunctuationFreeText')
        .asFunction();

    sherpaOnnxCreateAudioTagging ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateAudioTaggingNative>>(
            'SherpaOnnxCreateAudioTagging')
        .asFunction();

    sherpaOnnxDestroyAudioTagging ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyAudioTaggingNative>>(
            'SherpaOnnxDestroyAudioTagging')
        .asFunction();

    sherpaOnnxAudioTaggingCreateOfflineStream ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxAudioTaggingCreateOfflineStreamNative>>(
            'SherpaOnnxAudioTaggingCreateOfflineStream')
        .asFunction();

    sherpaOnnxAudioTaggingCompute ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxAudioTaggingComputeNative>>(
            'SherpaOnnxAudioTaggingCompute')
        .asFunction();

    sherpaOnnxAudioTaggingFreeResults ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxAudioTaggingFreeResultsNative>>(
            'SherpaOnnxAudioTaggingFreeResults')
        .asFunction();

    createKeywordSpotter ??= dynamicLibrary
        .lookup<NativeFunction<CreateKeywordSpotterNative>>(
            'SherpaOnnxCreateKeywordSpotter')
        .asFunction();

    destroyKeywordSpotter ??= dynamicLibrary
        .lookup<NativeFunction<DestroyKeywordSpotterNative>>(
            'SherpaOnnxDestroyKeywordSpotter')
        .asFunction();

    createKeywordStream ??= dynamicLibrary
        .lookup<NativeFunction<CreateKeywordStreamNative>>(
            'SherpaOnnxCreateKeywordStream')
        .asFunction();

    createKeywordStreamWithKeywords ??= dynamicLibrary
        .lookup<NativeFunction<CreateKeywordStreamWithKeywordsNative>>(
            'SherpaOnnxCreateKeywordStreamWithKeywords')
        .asFunction();

    isKeywordStreamReady ??= dynamicLibrary
        .lookup<NativeFunction<IsKeywordStreamReadyNative>>(
            'SherpaOnnxIsKeywordStreamReady')
        .asFunction();

    decodeKeywordStream ??= dynamicLibrary
        .lookup<NativeFunction<DecodeKeywordStreamNative>>(
            'SherpaOnnxDecodeKeywordStream')
        .asFunction();

    resetKeywordStream ??= dynamicLibrary
        .lookup<NativeFunction<ResetKeywordStreamNative>>(
            'SherpaOnnxResetKeywordStream')
        .asFunction();

    getKeywordResultAsJson ??= dynamicLibrary
        .lookup<NativeFunction<GetKeywordResultAsJsonNative>>(
            'SherpaOnnxGetKeywordResultAsJson')
        .asFunction();

    freeKeywordResultJson ??= dynamicLibrary
        .lookup<NativeFunction<FreeKeywordResultJsonNative>>(
            'SherpaOnnxFreeKeywordResultJson')
        .asFunction();

    createOfflineTts ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOfflineTtsNative>>(
            'SherpaOnnxCreateOfflineTts')
        .asFunction();

    destroyOfflineTts ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyOfflineTtsNative>>(
            'SherpaOnnxDestroyOfflineTts')
        .asFunction();

    offlineTtsSampleRate ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOfflineTtsSampleRateNative>>(
            'SherpaOnnxOfflineTtsSampleRate')
        .asFunction();

    offlineTtsNumSpeakers ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOfflineTtsNumSpeakersNative>>(
            'SherpaOnnxOfflineTtsNumSpeakers')
        .asFunction();

    offlineTtsGenerate ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOfflineTtsGenerateNative>>(
            'SherpaOnnxOfflineTtsGenerate')
        .asFunction();

    destroyOfflineTtsGeneratedAudio ??= dynamicLibrary
        .lookup<
                NativeFunction<
                    SherpaOnnxDestroyOfflineTtsGeneratedAudioNative>>(
            'SherpaOnnxDestroyOfflineTtsGeneratedAudio')
        .asFunction();

    offlineTtsGenerateWithCallback ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxOfflineTtsGenerateWithCallbackNative>>(
            'SherpaOnnxOfflineTtsGenerateWithCallback')
        .asFunction();

    createOfflineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<CreateOfflineRecognizerNative>>(
            'SherpaOnnxCreateOfflineRecognizer')
        .asFunction();

    destroyOfflineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOfflineRecognizerNative>>(
            'SherpaOnnxDestroyOfflineRecognizer')
        .asFunction();

    offlineRecognizerSetConfig ??= dynamicLibrary
        .lookup<NativeFunction<OfflineRecognizerSetConfigNative>>(
            'SherpaOnnxOfflineRecognizerSetConfig')
        .asFunction();

    createOfflineStream ??= dynamicLibrary
        .lookup<NativeFunction<CreateOfflineStreamNative>>(
            'SherpaOnnxCreateOfflineStream')
        .asFunction();

    destroyOfflineStream ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOfflineStreamNative>>(
            'SherpaOnnxDestroyOfflineStream')
        .asFunction();

    acceptWaveformOffline ??= dynamicLibrary
        .lookup<NativeFunction<AcceptWaveformOfflineNative>>(
            'SherpaOnnxAcceptWaveformOffline')
        .asFunction();

    decodeOfflineStream ??= dynamicLibrary
        .lookup<NativeFunction<DecodeOfflineStreamNative>>(
            'SherpaOnnxDecodeOfflineStream')
        .asFunction();

    getOfflineStreamResultAsJson ??= dynamicLibrary
        .lookup<NativeFunction<GetOfflineStreamResultAsJsonNative>>(
            'SherpaOnnxGetOfflineStreamResultAsJson')
        .asFunction();

    destroyOfflineStreamResultJson ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOfflineStreamResultJsonNative>>(
            'SherpaOnnxDestroyOfflineStreamResultJson')
        .asFunction();

    createOnlineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOnlineRecognizerNative>>(
            'SherpaOnnxCreateOnlineRecognizer')
        .asFunction();

    destroyOnlineRecognizer ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDestroyOnlineRecognizerNative>>(
            'SherpaOnnxDestroyOnlineRecognizer')
        .asFunction();

    createOnlineStream ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOnlineStreamNative>>(
            'SherpaOnnxCreateOnlineStream')
        .asFunction();

    createOnlineStreamWithHotwords ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxCreateOnlineStreamWithHotwordsNative>>(
            'SherpaOnnxCreateOnlineStreamWithHotwords')
        .asFunction();

    isOnlineStreamReady ??= dynamicLibrary
        .lookup<NativeFunction<IsOnlineStreamReadyNative>>(
            'SherpaOnnxIsOnlineStreamReady')
        .asFunction();

    decodeOnlineStream ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxDecodeOnlineStreamNative>>(
            'SherpaOnnxDecodeOnlineStream')
        .asFunction();

    getOnlineStreamResultAsJson ??= dynamicLibrary
        .lookup<NativeFunction<GetOnlineStreamResultAsJsonNative>>(
            'SherpaOnnxGetOnlineStreamResultAsJson')
        .asFunction();

    reset ??= dynamicLibrary
        .lookup<NativeFunction<ResetNative>>('SherpaOnnxOnlineStreamReset')
        .asFunction();

    isEndpoint ??= dynamicLibrary
        .lookup<NativeFunction<IsEndpointNative>>(
            'SherpaOnnxOnlineStreamIsEndpoint')
        .asFunction();

    destroyOnlineStreamResultJson ??= dynamicLibrary
        .lookup<NativeFunction<DestroyOnlineStreamResultJsonNative>>(
            'SherpaOnnxDestroyOnlineStreamResultJson')
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

    voiceActivityDetectorFlush ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxVoiceActivityDetectorFlushNative>>(
            'SherpaOnnxVoiceActivityDetectorFlush')
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
        .lookup<NativeFunction<SherpaOnnxDestroyOnlineStreamNative>>(
            'SherpaOnnxDestroyOnlineStream')
        .asFunction();

    onlineStreamAcceptWaveform ??= dynamicLibrary
        .lookup<NativeFunction<OnlineStreamAcceptWaveformNative>>(
            'SherpaOnnxOnlineStreamAcceptWaveform')
        .asFunction();

    onlineStreamInputFinished ??= dynamicLibrary
        .lookup<NativeFunction<OnlineStreamInputFinishedNative>>(
            'SherpaOnnxOnlineStreamInputFinished')
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

    getVersionStr ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxGetVersionStrNative>>(
            'SherpaOnnxGetVersionStr')
        .asFunction();

    getGitSha1 ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxGetGitSha1Native>>(
            'SherpaOnnxGetGitSha1')
        .asFunction();

    getGitDate ??= dynamicLibrary
        .lookup<NativeFunction<SherpaOnnxGetGitDateNative>>(
            'SherpaOnnxGetGitDate')
        .asFunction();
  }
}
