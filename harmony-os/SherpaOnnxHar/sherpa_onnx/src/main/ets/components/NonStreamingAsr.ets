import {
  acceptWaveformOffline,
  createOfflineRecognizer,
  createOfflineStream,
  decodeOfflineStream,
  getOfflineStreamResultAsJson,
  offlineRecognizerSetConfig,
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

export class HomophoneReplacerConfig {
  public dictDir: string = '';  // unused
  public lexicon: string = '';
  public ruleFsts: string = '';
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

export class OfflineDolphinModelConfig {
  public model: string = '';
}

export class OfflineOmnilingualAsrCtcModelConfig {
  public model: string = '';
}

export class OfflineMedAsrCtcModelConfig {
  public model: string = '';
}

export class OfflineFunASRNanoModelConfig {
  public encoderAdaptor: string = '';
  public llm: string = '';
  public embedding: string = '';
  public tokenizer: string = '';
  public systemPrompt: string = '';
  public userPrompt: string = '';
  public maxNewTokens: number = 0;
  public temperature: number = 1e-6;
  public topP: number = 0.8;
  public seed: number = 0;
  public language: string = '';
  public itn: number = 0;
  public hotwords: string = '';
}

export class OfflineQwen3AsrModelConfig {
  public convFrontend: string = '';
  public encoder: string = '';
  public decoder: string = '';
  public tokenizer: string = '';
  public maxTotalLen: number = 512;
  public maxNewTokens: number = 64;
  public temperature: number = 1e-6;
  public topP: number = 0.8;
  public seed: number = 42;
  public hotwords: string = '';
}

export class OfflineZipformerCtcModelConfig {
  public model: string = '';
}

export class OfflineWenetCtcModelConfig {
  public model: string = '';
}

export class OfflineFireRedAsrModelConfig {
  public encoder: string = '';
  public decoder: string = '';
}

export class OfflineFireRedAsrCtcModelConfig {
  public model: string = '';
}

export class OfflineWhisperModelConfig {
  public encoder: string = '';
  public decoder: string = '';
  public language: string = '';
  public task: string = 'transcribe';
  public tailPaddings: number = -1;
  public enableTokenTimestamps: boolean = false;
  public enableSegmentTimestamps: boolean = false;
}

export class OfflineCanaryModelConfig {
  public encoder: string = '';
  public decoder: string = '';
  public srcLang: string = '';
  public tgtLang: string = '';
  public usePnc: number = 1;
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
  public mergedDecoder: string = '';
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
  public fireRedAsr: OfflineFireRedAsrModelConfig = new OfflineFireRedAsrModelConfig();
  public dolphin: OfflineDolphinModelConfig = new OfflineDolphinModelConfig();
  public zipformerCtc: OfflineZipformerCtcModelConfig = new OfflineZipformerCtcModelConfig();
  public canary: OfflineCanaryModelConfig = new OfflineCanaryModelConfig();
  public wenetCtc: OfflineWenetCtcModelConfig = new OfflineWenetCtcModelConfig();
  public omnilingual: OfflineOmnilingualAsrCtcModelConfig = new OfflineOmnilingualAsrCtcModelConfig();
  public medasr: OfflineMedAsrCtcModelConfig = new OfflineMedAsrCtcModelConfig();
  public funasrNano: OfflineFunASRNanoModelConfig = new OfflineFunASRNanoModelConfig();
  public qwen3Asr: OfflineQwen3AsrModelConfig = new OfflineQwen3AsrModelConfig();
  public fireRedAsrCtc: OfflineFireRedAsrCtcModelConfig = new OfflineFireRedAsrCtcModelConfig();
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
  public hr: HomophoneReplacerConfig = new HomophoneReplacerConfig();
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

  setConfig(config: OfflineRecognizerConfig) {
    offlineRecognizerSetConfig(this.handle, config);
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
