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
 * Minimal offline model config used in JS-facing constructors.
 * @typedef {Object} OfflineModelConfig
 * @property {Object} [transducer] - {encoder?:string, decoder?:string, joiner?:string}
 * @property {Object} [paraformer] - {model?:string}
 * @property {string} [tokens]
 * @property {number} [numThreads]
 * @property {boolean|number} [debug]
 * @property {string} [provider]
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
 * Offline TTS configuration (partial, commonly used props).
 * @typedef {Object} OfflineTtsConfig
 * @property {Object} [model]
 * @property {number} [maxNumSentences]
 * @property {number} [silenceScale]
 * @property {number} [numThreads]
 * @property {string} [provider]
 */

/**
 * Offline Speech Denoiser configuration (partial).
 * @typedef {Object} OfflineSpeechDenoiserConfig
 * @property {Object} [model]
 * @property {number} [numThreads]
 * @property {string} [provider]
 */

/**
 * Offline Speaker Diarization configuration (partial).
 * @typedef {Object} OfflineSpeakerDiarizationConfig
 * @property {Object} [segmentation]
 * @property {Object} [embedding]
 * @property {Object} [clustering]
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
