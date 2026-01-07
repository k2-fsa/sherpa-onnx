/**
 * Centralized JSDoc typedefs for the Node addon API.
 * These typedefs mirror the shapes produced/consumed by the C++ bindings
 * in `scripts/node-addon-api/src/*` and by the underlying SherpaOnnx C API.
 *
 * Keep these typedefs specialized (no `any`/`unknown`) and concise.
 */

/**
 * Opaque handle types returned by native constructors. These are opaque
 * JavaScript objects backed by native pointers. Do not introspect or
 * mutate their internals; pass them to the API functions as-is.
 *
 * @typedef {Object} OfflineStreamHandle
 * @see src/non-streaming-asr.cc
 */

/**
 * @typedef {Object} OnlineStreamHandle
 * @see src/streaming-asr.cc
 */

/**
 * @typedef {Object} OfflineRecognizerHandle
 * @see src/non-streaming-asr.cc
 */

/**
 * @typedef {Object} OnlineRecognizerHandle
 * @see src/streaming-asr.cc
 */

/**
 * @typedef {Object} DisplayHandle
 * @see src/streaming-asr.cc
 */

/**
 * @typedef {Object} CircularBufferHandle
 * @see src/vad.cc
 */

/**
 * @typedef {Object} VoiceActivityDetectorHandle
 * @see src/vad.cc
 */

/**
 * @typedef {Object} AudioTaggingHandle
 * @see src/audio-tagging.cc
 */

/**
 * @typedef {Object} OfflinePunctuationHandle
 * @see src/punctuation.cc
 */

/**
 * A single audio event returned by AudioTagging.compute().
 * @typedef {Object} AudioEvent
 * @property {string} name - The event name.
 * @property {number} prob - Probability in [0,1].
 * @property {number} index - Index (integer) of the event.
 */

/**
 * AudioTagging specific model config for Zipformer variant
 * @typedef {Object} AudioTaggingZipformerModelConfig
 * @property {string} [model]
 */

/**
 * AudioTagging model config.
 * @typedef {Object} AudioTaggingModelConfig
 * @property {AudioTaggingZipformerModelConfig} [zipformer]
 * @property {string} [ced]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * AudioTagging configuration passed to constructor.
 * @typedef {Object} AudioTaggingConfig
 * @property {AudioTaggingModelConfig} [model]
 * @property {string} [labels]
 * @property {number} [topK]
 */

/**
 * Waveform input object used by acceptWaveform methods.
 * @typedef {Object} Waveform
 * @property {Float32Array} samples - Float32Array of samples in [-1, 1].
 * @property {number} sampleRate - Sample rate as an integer (e.g., 16000).
 */

/**
 * Feature config used by recognizers and models.
 * @typedef {Object} FeatureConfig
 * @property {number} [sampleRate]
 * @property {number} [featureDim]
 */

/**
 * Silero VAD model config
 * @typedef {Object} SileroVadModelConfig
 * @property {string} [model]
 * @property {number} [threshold]
 * @property {number} [minSilenceDuration]
 * @property {number} [minSpeechDuration]
 * @property {number} [windowSize]
 * @property {number} [maxSpeechDuration]
 */

/**
 * Ten-VAD model config
 * @typedef {Object} TenVadModelConfig
 * @property {string} [model]
 * @property {number} [threshold]
 * @property {number} [minSilenceDuration]
 * @property {number} [minSpeechDuration]
 * @property {number} [windowSize]
 * @property {number} [maxSpeechDuration]
 */

/**
 * Voice activity detector configuration.
 * @typedef {Object} VadConfig
 * @property {SileroVadModelConfig} [sileroVad]
 * @property {TenVadModelConfig} [tenVad]
 * @property {number} [sampleRate]
 * @property {number} [numThreads]
 * @property {string} [provider]
 * @property {boolean|number} [debug]
 */

/**
 * Offline Transducer model config
 * @typedef {Object} OfflineTransducerModelConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 * @property {string} [joiner]
 */

/**
 * Offline Paraformer model config
 * @typedef {Object} OfflineParaformerModelConfig
 * @property {string} [model]
 */

/**
 * Offline Zipformer CTC model config
 * @typedef {Object} OfflineZipformerCtcModelConfig
 * @property {string} [model]
 */

/**
 * Offline Wenet CTC model config
 * @typedef {Object} OfflineWenetCtcModelConfig
 * @property {string} [model]
 */

/**
 * Offline Omnilingual ASR CTC model config
 * @typedef {Object} OfflineOmnilingualAsrCtcModelConfig
 * @property {string} [model]
 */

/**
 * Offline Med ASR CTC model config
 * @typedef {Object} OfflineMedAsrCtcModelConfig
 * @property {string} [model]
 */

/**
 * Offline Dolphin model config
 * @typedef {Object} OfflineDolphinModelConfig
 * @property {string} [model]
 */

/**
 * Offline NeMo CTC model config
 * @typedef {Object} OfflineNeMoCtcModelConfig
 * @property {string} [model]
 */

/**
 * Offline Canary model config
 * @typedef {Object} OfflineCanaryModelConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 * @property {string} [srcLang]
 * @property {string} [tgtLang]
 * @property {number} [usePnc]
 */

/**
 * Offline Whisper model config
 * @typedef {Object} OfflineWhisperModelConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 * @property {string} [language]
 * @property {string} [task]
 * @property {number} [tailPaddings]
 */

/**
 * Offline FireRed ASR model config
 * @typedef {Object} OfflineFireRedAsrModelConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 */

/**
 * Offline Moonshine model config
 * @typedef {Object} OfflineMoonshineModelConfig
 * @property {string} [preprocessor]
 * @property {string} [encoder]
 * @property {string} [uncachedDecoder]
 * @property {string} [cachedDecoder]
 */

/**
 * Offline TDNN model config
 * @typedef {Object} OfflineTdnnModelConfig
 * @property {string} [model]
 */

/**
 * Offline SenseVoice model config
 * @typedef {Object} OfflineSenseVoiceModelConfig
 * @property {string} [model]
 * @property {string} [language]
 * @property {number} [useInverseTextNormalization]
 */

/**
 * Offline model config.
 * @typedef {Object} OfflineModelConfig
 * @property {OfflineTransducerModelConfig} [transducer]
 * @property {OfflineParaformerModelConfig} [paraformer]
 * @property {OfflineZipformerCtcModelConfig} [zipformerCtc]
 * @property {OfflineWenetCtcModelConfig} [wenetCtc]
 * @property {OfflineOmnilingualAsrCtcModelConfig} [omnilingual]
 * @property {OfflineMedAsrCtcModelConfig} [medasr]
 * @property {OfflineDolphinModelConfig} [dolphin]
 * @property {OfflineNeMoCtcModelConfig} [nemoCtc]
 * @property {OfflineCanaryModelConfig} [canary]
 * @property {OfflineWhisperModelConfig} [whisper]
 * @property {OfflineFireRedAsrModelConfig} [fireRedAsr]
 * @property {OfflineMoonshineModelConfig} [moonshine]
 * @property {OfflineTdnnModelConfig} [tdnn]
 * @property {OfflineSenseVoiceModelConfig} [senseVoice]
 * @property {string} [tokens]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * Transducer model config
 * @typedef {Object} TransducerModelConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 * @property {string} [joiner]
 */

/**
 * Paraformer model config
 * @typedef {Object} ParaformerModelConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 */

/**
 * Zipformer2 CTC model config
 * @typedef {Object} Zipformer2CtcModelConfig
 * @property {string} [model]
 */

/**
 * NeMo CTC model config
 * @typedef {Object} NemoCtcModelConfig
 * @property {string} [model]
 */

/**
 * Tone CTC model config
 * @typedef {Object} ToneCtcModelConfig
 * @property {string} [model]
 */

/**
 * Online model config (subset of C++ `OnlineModelConfig`).
 * @typedef {Object} OnlineModelConfig
 * @property {TransducerModelConfig} [transducer]
 * @property {ParaformerModelConfig} [paraformer]
 * @property {Zipformer2CtcModelConfig} [zipformer2Ctc]
 * @property {NemoCtcModelConfig} [nemoCtc]
 * @property {ToneCtcModelConfig} [toneCtc]
 * @property {string} [tokens]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 * @property {string} [modelType]
 * @property {string} [modelingUnit]
 * @property {string} [bpeVocab]
 * @property {string} [tokensBuf]
 * @property {number} [tokensBufSize]
 */

/**
 * Homophone replacer configuration used both in online and offline recognizers.
 * @typedef {Object} HomophoneReplacerConfig
 * @property {string} [lexicon]
 * @property {string} [ruleFsts]
 */

/**
 * Online recognizer configuration passed to createOnlineRecognizer.
 * @typedef {Object} OnlineRecognizerConfig
 * @property {FeatureConfig} [featConfig]
 * @property {OnlineModelConfig} [modelConfig]
 * @property {HomophoneReplacerConfig} [hr]
 * @property {string} [decodingMethod]
 * @property {number} [maxActivePaths]
 * @property {boolean|number} [enableEndpoint]
 * @property {number} [rule1MinTrailingSilence]
 * @property {number} [rule2MinTrailingSilence]
 * @property {number} [rule3MinUtteranceLength]
 * @property {string} [hotwordsFile]
 * @property {number} [hotwordsScore]
 * @property {string} [ruleFsts]
 * @property {string} [ruleFars]
 * @property {number} [blankPenalty]
 */

/**
 * Offline recognizer config passed to createOfflineRecognizer.
 * @typedef {Object} OfflineRecognizerConfig
 * @property {FeatureConfig} [featConfig]
 * @property {OfflineModelConfig} [modelConfig]
 */

/**
 * Wave object returned by readWave and used by writeWave.
 * @typedef {Object} WaveObject
 * @property {Float32Array} samples - 1-D float32 samples in [-1, 1].
 * @property {number} sampleRate - Sample rate as an integer (e.g., 16000).
 * @see src/wave-reader.cc
 */

/**
 * Speech segment returned by Vad.front().
 * @typedef {Object} SpeechSegment
 * @property {number} start - Start index (int32) of this segment.
 * @property {Float32Array} samples - Float32Array of samples.
 * @see src/vad.cc
 */

/**
 * Audio returned by TTS and speech denoiser.
 * @typedef {Object} GeneratedAudio
 * @property {Float32Array} samples - The generated/denoised audio samples.
 * @property {number} sampleRate - Sample rate in Hz.
 * @see src/non-streaming-tts.cc
 * @see src/non-streaming-speech-denoiser.cc
 */

/**
 * TTS request object passed to generate/generateAsync.
 * @typedef {Object} TtsRequest
 * @property {string} text - Input text to synthesize.
 * @property {number} sid - Speaker id (integer).
 * @property {number} speed - Playback speed (float).
 * @property {boolean} [enableExternalBuffer=true] - Whether to use an external buffer.
 */

/**
 * Spoken Language ID whisper config
 * @typedef {Object} SpokenLanguageIdentificationWhisperConfig
 * @property {string} [encoder]
 * @property {string} [decoder]
 * @property {number} [tailPaddings]
 */

/**
 * SpokenLanguageIdentification config
 * @typedef {Object} SpokenLanguageIdentificationConfig
 * @property {SpokenLanguageIdentificationWhisperConfig} [whisper]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * Speaker embedding extractor config
 * @typedef {Object} SpeakerEmbeddingExtractorConfig
 * @property {string} [model]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * Offline punctuation model config
 * @typedef {Object} OfflinePunctuationModelConfig
 * @property {string} [ctTransformer]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * Offline punctuation config
 * @typedef {Object} OfflinePunctuationConfig
 * @property {OfflinePunctuationModelConfig} [model]
 */

/**
 * Online punctuation model config
 * @typedef {Object} OnlinePunctuationModelConfig
 * @property {string} [cnnBilstm]
 * @property {string} [bpeVocab]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * Online punctuation config
 * @typedef {Object} OnlinePunctuationConfig
 * @property {OnlinePunctuationModelConfig} [model]
 */

/**
 * Generic audio processing request used by denoisers/tts generators.
 * @typedef {Object} AudioProcessRequest
 * @property {Float32Array} samples
 * @property {number} sampleRate
 * @property {boolean} [enableExternalBuffer]
 */

/**
 * Offline TTS model configs
 * @typedef {Object} OfflineTtsVitsModelConfig
 * @property {string} [model]
 * @property {string} [lexicon]
 * @property {string} [tokens]
 * @property {string} [dataDir]
 * @property {number} [noiseScale]
 * @property {number} [noiseScaleW]
 * @property {number} [lengthScale]
 */

/**
 * @typedef {Object} OfflineTtsMatchaModelConfig
 * @property {string} [acousticModel]
 * @property {string} [vocoder]
 * @property {string} [lexicon]
 * @property {string} [tokens]
 * @property {string} [dataDir]
 * @property {number} [noiseScale]
 * @property {number} [lengthScale]
 */

/**
 * @typedef {Object} OfflineTtsKokoroModelConfig
 * @property {string} [model]
 * @property {string} [voices]
 * @property {string} [tokens]
 * @property {string} [dataDir]
 * @property {number} [lengthScale]
 * @property {string} [lexicon]
 * @property {string} [lang]
 */

/**
 * @typedef {Object} OfflineTtsKittenModelConfig
 * @property {string} [model]
 * @property {string} [voices]
 * @property {string} [tokens]
 * @property {string} [dataDir]
 * @property {number} [lengthScale]
 */

/**
 * Offline TTS model config
 * @typedef {Object} OfflineTtsModelConfig
 * @property {OfflineTtsVitsModelConfig} [vits]
 * @property {OfflineTtsMatchaModelConfig} [matcha]
 * @property {OfflineTtsKokoroModelConfig} [kokoro]
 * @property {OfflineTtsKittenModelConfig} [kitten]
 */

/**
 * Offline TTS configuration (partial, commonly used props).
 * @typedef {Object} OfflineTtsConfig
 * @property {OfflineTtsModelConfig} [model]
 * @property {number} [maxNumSentences]
 * @property {number} [silenceScale]
 * @property {number} [numThreads]
 * @property {string} [provider]
 */

/**
 * Offline Speech Denoiser model config
 * @typedef {Object} OfflineSpeechDenoiserGtcrnModelConfig
 * @property {string} [model]
 */

/**
 * Offline Speech Denoiser model config
 * @typedef {Object} OfflineSpeechDenoiserModelConfig
 * @property {OfflineSpeechDenoiserGtcrnModelConfig} [gtcrn]
 */

/**
 * Offline Speech Denoiser configuration (partial).
 * @typedef {Object} OfflineSpeechDenoiserConfig
 * @property {OfflineSpeechDenoiserModelConfig} [model]
 * @property {number} [numThreads]
 * @property {string} [provider]
 */

/**
 * Offline speaker segmentation (pyannote) model config
 * @typedef {Object} OfflineSpeakerSegmentationPyannoteModelConfig
 * @property {string} [model]
 */

/**
 * Offline speaker segmentation model config
 * @typedef {Object} OfflineSpeakerSegmentationModelConfig
 * @property {OfflineSpeakerSegmentationPyannoteModelConfig} [pyannote]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
 */

/**
 * Offline Speaker Diarization configuration (partial).
 * @typedef {Object} OfflineSpeakerDiarizationConfig
 * @property {OfflineSpeakerSegmentationModelConfig} [segmentation]
 * @property {SpeakerEmbeddingExtractorConfig} [embedding]
 * @property {FastClusteringConfig} [clustering]
 * @property {number} [minDurationOn]
 * @property {number} [minDurationOff]
 */

/**
 * Fast clustering configuration used by diarization.
 * @typedef {Object} FastClusteringConfig
 * @property {number} [numClusters]
 * @property {number} [threshold]
 */

/**
 * SpeakerEmbeddingManager add-multi flattened object
 * @typedef {Object} SpeakerEmbeddingManagerAddListFlattenedObj
 * @property {string} name
 * @property {Float32Array} vv
 * @property {number} n
 */

/**
 * SpeakerEmbeddingManager search object
 * @typedef {Object} SpeakerEmbeddingManagerSearchObj
 * @property {Float32Array} v
 * @property {number} threshold
 */

/**
 * SpeakerEmbeddingManager verify object
 * @typedef {Object} SpeakerEmbeddingManagerVerifyObj
 * @property {string} name
 * @property {Float32Array} v
 * @property {number} threshold
 */

/**
 * KeywordSpotter config (partial)
 * @typedef {Object} KeywordSpotterConfig
 * @property {FeatureConfig} [featConfig]
 * @property {OfflineModelConfig} [modelConfig]
 * @property {number} [maxActivePaths]
 * @property {number} [numTrailingBlanks]
 * @property {number} [keywordsScore]
 * @property {number} [keywordsThreshold]
 * @property {string} [keywordsFile]
 */

/**
 * Offline recognition result returned by `getOfflineStreamResultAsJson`.
 * See `OfflineRecognitionResult::AsJsonString()` in C++ for precise fields.
 * @typedef {Object} OfflineRecognizerResult
 * @property {string} lang
 * @property {string} emotion
 * @property {string} event
 * @property {string} text
 * @property {number[]} timestamps
 * @property {number[]} durations
 * @property {string[]} tokens
 * @property {number[]} ys_log_probs
 * @property {number[]} words
 */

/**
 * Online recognition result returned by `getOnlineStreamResultAsJson`.
 * See `OnlineRecognizerResult::AsJsonString()` in C++.
 * @typedef {Object} OnlineRecognizerResult
 * @property {string} text
 * @property {string[]} tokens
 * @property {number[]} timestamps
 * @property {number[]} ys_probs
 * @property {number[]} lm_probs
 * @property {number[]} context_scores
 * @property {number} segment
 * @property {number[]} words
 * @property {number} start_time
 * @property {boolean} is_final
 * @property {boolean} is_eof
 */

/**
 * Keyword spotter result returned by `getKeywordResultAsJson`.
 * @typedef {Object} KeywordResult
 * @property {number} start_time
 * @property {string} keyword
 * @property {number[]} timestamps
 * @property {string[]} tokens
 */

/**
 * Speaker diarization segment returned by `offlineSpeakerDiarizationProcess`.
 * @typedef {Object} SpeakerDiarizationSegment
 * @property {number} start - start time in seconds
 * @property {number} end - end time in seconds
 * @property {number} speaker - speaker id (integer)
 */

/**
 * Speaker embedding entry used by SpeakerEmbeddingManager.add
 * @typedef {Object} SpeakerEmbeddingEntry
 * @property {string} name - speaker name
 * @property {Float32Array} v - embedding vector
 */

/**
 * @typedef {Object} OnlineStreamObject
 * @property {OnlineStreamHandle} handle
 */

/**
 * @typedef {Object} DisplayObject
 * @property {DisplayHandle} handle
 */

// Export typedefs so they can be referenced by require('./types.js')
module.exports = {};
