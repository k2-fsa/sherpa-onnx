import {
  createOfflineTts,
  getOfflineTtsNumSpeakers,
  getOfflineTtsSampleRate,
  offlineTtsGenerate,
  offlineTtsGenerateAsyncWithConfig,
  offlineTtsGenerateAsync,
  offlineTtsGenerateWithConfig,
} from 'libsherpa_onnx.so';

export class OfflineTtsVitsModelConfig {
  public model: string = '';
  public lexicon: string = '';
  public tokens: string = '';
  public dataDir: string = '';
  public dictDir: String = '';  // unused
  public noiseScale: number = 0.667;
  public noiseScaleW: number = 0.8;
  public lengthScale: number = 1.0;
}

export class OfflineTtsMatchaModelConfig {
  public acousticModel: string = '';
  public vocoder: string = '';
  public lexicon: string = '';
  public tokens: string = '';
  public dataDir: string = '';
  public dictDir: String = '';  // unused
  public noiseScale: number = 0.667;
  public lengthScale: number = 1.0;
}

export class OfflineTtsKokoroModelConfig {
  public model: string = '';
  public voices: string = '';
  public tokens: string = '';
  public dataDir: string = '';
  public lengthScale: number = 1.0;
  public dictDir: string = '';  // unused
  public lexicon: string = '';
  public lang: string = '';
}

export class OfflineTtsKittenModelConfig {
  public model: string = '';
  public voices: string = '';
  public tokens: string = '';
  public dataDir: string = '';
  public lengthScale: number = 1.0;
}

export class OfflineTtsZipvoiceModelConfig {
  public tokens: string = '';
  public encoder: string = '';
  public decoder: string = '';
  public vocoder: string = '';
  public dataDir: string = '';
  public lexicon: string = '';
  public featScale: number = 0.1;
  public tShift: number = 0.5;
  public targetRms: number = 0.1;
  public guidanceScale: number = 1.0;
}

export class OfflineTtsPocketModelConfig {
  public lmFlow: string = '';
  public lmMain: string = '';
  public encoder: string = '';
  public decoder: string = '';
  public textConditioner: string = '';
  public vocabJson: string = '';
  public tokenScoresJson: string = '';
  public voiceEmbeddingCacheCapacity: number = 50;
}

export class OfflineTtsSupertonicModelConfig {
  public durationPredictor: string = '';
  public textEncoder: string = '';
  public vectorEstimator: string = '';
  public vocoder: string = '';
  public ttsJson: string = '';
  public unicodeIndexer: string = '';
  public voiceStyle: string = '';
}

export class OfflineTtsModelConfig {
  public vits: OfflineTtsVitsModelConfig = new OfflineTtsVitsModelConfig();
  public matcha: OfflineTtsMatchaModelConfig = new OfflineTtsMatchaModelConfig();
  public kokoro: OfflineTtsKokoroModelConfig = new OfflineTtsKokoroModelConfig();
  public kitten: OfflineTtsKittenModelConfig = new OfflineTtsKittenModelConfig();
  public zipvoice: OfflineTtsZipvoiceModelConfig = new OfflineTtsZipvoiceModelConfig();
  public pocket: OfflineTtsPocketModelConfig = new OfflineTtsPocketModelConfig();
  public supertonic: OfflineTtsSupertonicModelConfig = new OfflineTtsSupertonicModelConfig();
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OfflineTtsConfig {
  public model: OfflineTtsModelConfig = new OfflineTtsModelConfig();
  public ruleFsts: string = '';
  public ruleFars: string = '';
  public maxNumSentences: number = 1;
  public silenceScale: number = 0.2;
}

export class TtsOutput {
  public samples: Float32Array = new Float32Array(0);
  public sampleRate: number = 0;
}

interface TtsCallbackData {
  samples: Float32Array;
  progress: number;
}

export class TtsGenerationConfig {
  public silenceScale: number = 0.2;
  public speed: number = 1.0;
  public sid: number = 0;
  public referenceAudio?: Float32Array;
  public referenceSampleRate: number = 0;
  public referenceText: string = '';
  public numSteps: number = 5;
  public extra: object = new Object();
}

export class TtsInput {
  public text: string = '';
  public sid: number = 0;
  public speed: number = 1.0;
  public enableExternalBuffer: boolean = true;
  public callback?: (data: TtsCallbackData) => number;
}

export class TtsInputWithConfig {
  public text: string = '';
  public generationConfig: TtsGenerationConfig = new TtsGenerationConfig();
  public enableExternalBuffer: boolean = true;
  public callback?: (data: TtsCallbackData) => number;
}

export class OfflineTts {
  public config: OfflineTtsConfig;
  public numSpeakers: number;
  public sampleRate: number;
  private handle: object;

  constructor(config: OfflineTtsConfig, mgr?: object) {
    this.handle = createOfflineTts(config, mgr);
    this.config = config;

    this.numSpeakers = getOfflineTtsNumSpeakers(this.handle);
    this.sampleRate = getOfflineTtsSampleRate(this.handle);
  }

  /*
   input obj: {text: "xxxx", sid: 0, speed: 1.0}
   where text is a string, sid is a int32, speed is a float

   return an object {samples: Float32Array, sampleRate: <a number>}
   */
  generate(input: TtsInput): TtsOutput {
    return offlineTtsGenerate(this.handle, input);
  }

  generateAsync(input: TtsInput): Promise<TtsOutput> {
    return offlineTtsGenerateAsync(this.handle, input);
  }

  generateWithConfig(input: TtsInputWithConfig): TtsOutput {
    return offlineTtsGenerateWithConfig(this.handle, input);
  }

  generateAsyncWithConfig(input: TtsInputWithConfig): Promise<TtsOutput> {
    return offlineTtsGenerateAsyncWithConfig(this.handle, input);
  }
}
