export const listRawfileDir: (mgr: object, dir: string) => Array<string>;

export const readWave: (filename: string, enableExternalBuffer: boolean = true) => {samples: Float32Array, sampleRate: number};
export const readWaveFromBinary: (data: Uint8Array, enableExternalBuffer: boolean = true) => {samples: Float32Array, sampleRate: number};
export const createCircularBuffer: (capacity: number) => object;
export const circularBufferPush: (handle: object, samples: Float32Array) => void;
export const circularBufferGet: (handle: object, index: number, n: number, enableExternalBuffer: boolean = true) => Float32Array;
export const circularBufferPop: (handle: object, n: number) => void;
export const circularBufferSize: (handle: object) => number;
export const circularBufferHead: (handle: object) => number;
export const circularBufferReset: (handle: object) => void;

export const createVoiceActivityDetector: (config: object, bufferSizeInSeconds: number, mgr?: object) => object;
export const voiceActivityDetectorAcceptWaveform: (handle: object, samples: Float32Array) => void;
export const voiceActivityDetectorIsEmpty: (handle: object) => boolean;
export const voiceActivityDetectorIsDetected: (handle: object) => boolean;
export const voiceActivityDetectorPop: (handle: object) => void;
export const voiceActivityDetectorClear: (handle: object) => void;
export const voiceActivityDetectorFront: (handle: object, enableExternalBuffer: boolean = true) => {samples: Float32Array, start: number};
export const voiceActivityDetectorReset: (handle: object) => void;
export const voiceActivityDetectorFlush: (handle: object) => void;

export const createOfflineRecognizer: (config: object, mgr?: object) => object;
export const createOfflineStream: (handle: object) => object;
export const acceptWaveformOffline: (handle: object, audio: object) => void;
export const decodeOfflineStream: (handle: object, streamHandle: object) => void;
export const getOfflineStreamResultAsJson: (streamHandle: object) => string;

export const createOnlineRecognizer: (config: object, mgr?: object) => object;
export const createOnlineStream: (handle: object) => object;
export const acceptWaveformOnline: (handle: object, audio: object) => void;
export const inputFinished: (streamHandle: object) => void;
export const isOnlineStreamReady: (handle: object, streamHandle: object) => boolean;
export const decodeOnlineStream: (handle: object, streamHandle: object) => void;
export const isEndpoint: (handle: object, streamHandle: object) => boolean;
export const reset: (handle: object, streamHandle: object) => void;
export const getOnlineStreamResultAsJson: (handle: object, streamHandle: object) => string;

export const createOfflineTts: (config: object, mgr?: object) => object;
export const getOfflineTtsNumSpeakers: (handle: object) => number;
export const getOfflineTtsSampleRate: (handle: object) => number;

export type TtsOutput = {
  samples: Float32Array;
  sampleRate: number;
};

export const offlineTtsGenerate: (handle: object, input: object) => TtsOutput;
export const offlineTtsGenerateAsync: (handle: object, input: object) => Promise<TtsOutput>;

export const createSpeakerEmbeddingExtractor: (config: object, mgr?: object) => object;
export const speakerEmbeddingExtractorDim: (handle: object) => number;
export const speakerEmbeddingExtractorCreateStream: (handle: object) => object;
export const speakerEmbeddingExtractorIsReady: (handle: object, stream: object) => boolean;
export const speakerEmbeddingExtractorComputeEmbedding: (handle: object, stream: object, enableExternalBuffer: boolean) => Float32Array;
export const createSpeakerEmbeddingManager: (dim: number) => object;
export const speakerEmbeddingManagerAdd: (handle: object, speaker: {name: string, v: Float32Array}) => boolean;
export const speakerEmbeddingManagerAddListFlattened: (handle: object, speaker: {name: string, vv: Float32Array, n: number}) => boolean;
export const speakerEmbeddingManagerRemove: (handle: object, name: string) => boolean;
export const speakerEmbeddingManagerSearch: (handle: object, obj: {v: Float32Array, threshold: number}) => string;
export const speakerEmbeddingManagerVerify: (handle: object, obj: {name: string, v: Float32Array, threshold: number}) => boolean;
export const speakerEmbeddingManagerContains: (handle: object, name: string) => boolean;
export const speakerEmbeddingManagerNumSpeakers: (handle: object) => number;
export const speakerEmbeddingManagerGetAllSpeakers: (handle: object) => Array<string>;

export const createOfflineSpeakerDiarization: (config: object, mgr?: object) => object;
export const getOfflineSpeakerDiarizationSampleRate: (handle: object) => number;
export const offlineSpeakerDiarizationProcess: (handle: object, input: object) => object;
export const offlineSpeakerDiarizationProcessAsync: (handle: object, input: object, callback: object) => object;
export const offlineSpeakerDiarizationSetConfig: (handle: object, config: object) => void;
