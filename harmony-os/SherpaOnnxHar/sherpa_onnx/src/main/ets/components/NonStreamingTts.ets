import {
  createOfflineTts,
  getOfflineTtsNumSpeakers,
  getOfflineTtsSampleRate,
  offlineTtsGenerate,
  offlineTtsGenerateAsync,
} from 'libsherpa_onnx.so';

export class OfflineTtsVitsModelConfig {
  public model: string = '';
  public lexicon: string = '';
  public tokens: string = '';
  public dataDir: string = '';
  public dictDir: String = '';
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
  public dictDir: String = '';
  public noiseScale: number = 0.667;
  public lengthScale: number = 1.0;
}

export class OfflineTtsModelConfig {
  public vits: OfflineTtsVitsModelConfig = new OfflineTtsVitsModelConfig();
  public matcha: OfflineTtsMatchaModelConfig = new OfflineTtsMatchaModelConfig();
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OfflineTtsConfig {
  public model: OfflineTtsModelConfig = new OfflineTtsModelConfig();
  public ruleFsts: string = '';
  public ruleFars: string = '';
  public maxNumSentences: number = 1;
}

export class TtsOutput {
  public samples: Float32Array = new Float32Array(0);
  public sampleRate: number = 0;
}

interface TtsCallbackData {
  samples: Float32Array;
  progress: number;
}

export class TtsInput {
  public text: string = '';
  public sid: number = 0;
  public speed: number = 1.0;
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
    return offlineTtsGenerate(this.handle, input) as TtsOutput;
  }

  generateAsync(input: TtsInput): Promise<TtsOutput> {
    return offlineTtsGenerateAsync(this.handle, input);
  }
}