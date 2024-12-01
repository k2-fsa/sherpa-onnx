import {
  acceptWaveformOffline,
  createOfflineRecognizer,
  createOfflineStream,
  decodeOfflineStream,
  getOfflineStreamResultAsJson,
} from 'libsherpa_onnx.so';

export interface Samples {
  samples: Float32Array;
  sampleRate: number;
}

export class OfflineStream {
  public handle: object;

  constructor(handle: object) {
    this.handle = handle;
  }

  // obj is {samples: samples, sampleRate: sampleRate}
  // samples is a float32 array containing samples in the range [-1, 1]
  // sampleRate is a number
  acceptWaveform(obj: Samples) {
    acceptWaveformOffline(this.handle, obj)
  }
}

export class FeatureConfig {
  public sampleRate: number = 16000;
  public featureDim: number = 80;
}

export class OfflineTransducerModelConfig {
  public encoder: string = '';
  public decoder: string = '';
  public joiner: string = '';
}

export class OfflineParaformerModelConfig {
  public model: string = '';
}

export class OfflineNemoEncDecCtcModelConfig {
  public model: string = '';
}

export class OfflineWhisperModelConfig {
  public encoder: string = '';
  public decoder: string = '';
  public language: string = '';
  public task: string = 'transcribe';
  public tailPaddings: number = -1;
}

export class OfflineTdnnModelConfig {
  public model: string = '';
}

export class OfflineSenseVoiceModelConfig {
  public model: string = '';
  public language: string = '';
  public useItn: boolean = false;
}

export class OfflineMoonshineModelConfig {
  public preprocessor: string = '';
  public encoder: string = '';
  public uncachedDecoder: string = '';
  public cachedDecoder: string = '';
}

export class OfflineModelConfig {
  public transducer: OfflineTransducerModelConfig = new OfflineTransducerModelConfig();
  public paraformer: OfflineParaformerModelConfig = new OfflineParaformerModelConfig();
  public nemoCtc: OfflineNemoEncDecCtcModelConfig = new OfflineNemoEncDecCtcModelConfig();
  public whisper: OfflineWhisperModelConfig = new OfflineWhisperModelConfig();
  public tdnn: OfflineTdnnModelConfig = new OfflineTdnnModelConfig();
  public tokens: string = '';
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
  public modelType: string = '';
  public modelingUnit: string = "cjkchar";
  public bpeVocab: string = '';
  public telespeechCtc: string = '';
  public senseVoice: OfflineSenseVoiceModelConfig = new OfflineSenseVoiceModelConfig();
  public moonshine: OfflineMoonshineModelConfig = new OfflineMoonshineModelConfig();
}

export class OfflineLMConfig {
  public model: string = '';
  public scale: number = 1.0;
}

export class OfflineRecognizerConfig {
  public featConfig: FeatureConfig = new FeatureConfig();
  public modelConfig: OfflineModelConfig = new OfflineModelConfig();
  public lmConfig: OfflineLMConfig = new OfflineLMConfig();
  public decodingMethod: string = "greedy_search";
  public maxActivePaths: number = 4;
  public hotwordsFfile: string = '';
  public hotwordsScore: number = 1.5;
  public ruleFsts: string = '';
  public ruleFars: string = '';
  public blankPenalty: number = 0;
}

export class OfflineRecognizerResult {
  public text: string = '';
  public timestamps: number[] = [];
  public tokens: string[] = [];
  public json = '';
  public lang: string = '';
  public emotion: string = '';
  public event: string = '';
}

interface OfflineRecognizerResultJson {
  text: string;
  timestamps: number[];
  tokens: string[];
  lang: string;
  emotion: string;
  event: string;
}

export class OfflineRecognizer {
  public handle: object;
  public config: OfflineRecognizerConfig;

  constructor(config: OfflineRecognizerConfig, mgr?: object) {
    this.handle = createOfflineRecognizer(config, mgr);
    this.config = config
  }

  createStream(): OfflineStream {
    const handle: object = createOfflineStream(this.handle);
    return new OfflineStream(handle);
  }

  decode(stream: OfflineStream) {
    decodeOfflineStream(this.handle, stream.handle);
  }

  getResult(stream: OfflineStream): OfflineRecognizerResult {
    const jsonStr: string = getOfflineStreamResultAsJson(stream.handle);

    let o = JSON.parse(jsonStr) as OfflineRecognizerResultJson;

    const r = new OfflineRecognizerResult()
    r.text = o.text
    r.timestamps = o.timestamps;
    r.tokens = o.tokens;
    r.json = jsonStr;
    r.lang = o.lang;
    r.emotion = o.emotion;
    r.event = o.event;

    return r;
  }
}