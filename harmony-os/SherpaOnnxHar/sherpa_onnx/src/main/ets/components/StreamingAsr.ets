import {
  acceptWaveformOnline,
  createOnlineRecognizer,
  createOnlineStream,
  decodeOnlineStream,
  getOnlineStreamResultAsJson,
  inputFinished,
  isEndpoint,
  isOnlineStreamReady,
  reset,
} from 'libsherpa_onnx.so';

import { FeatureConfig, Samples } from './NonStreamingAsr';

export class OnlineStream {
  public handle: object;

  constructor(handle: object) {
    this.handle = handle;
  }

  // obj is {samples: samples, sampleRate: sampleRate}
  // samples is a float32 array containing samples in the range [-1, 1]
  // sampleRate is a number
  acceptWaveform(obj: Samples) {
    acceptWaveformOnline(this.handle, obj)
  }

  inputFinished() {
    inputFinished(this.handle)
  }
}

export class OnlineTransducerModelConfig {
  public encoder: string = '';
  public decoder: string = '';
  public joiner: string = '';
}

export class OnlineParaformerModelConfig {
  public encoder: string = '';
  public decoder: string = '';
}

export class OnlineZipformer2CtcModelConfig {
  public model: string = '';
}

export class OnlineModelConfig {
  public transducer: OnlineTransducerModelConfig = new OnlineTransducerModelConfig();
  public paraformer: OnlineParaformerModelConfig = new OnlineParaformerModelConfig();
  public zipformer2_ctc: OnlineZipformer2CtcModelConfig = new OnlineZipformer2CtcModelConfig();
  public tokens: string = '';
  public numThreads: number = 1;
  public provider: string = 'cpu';
  public debug: boolean = false;
  public modelType: string = '';
  public modelingUnit: string = "cjkchar";
  public bpeVocab: string = '';
}

export class OnlineCtcFstDecoderConfig {
  public graph: string = '';
  public maxActive: number = 3000;
}

export class OnlineRecognizerConfig {
  public featConfig: FeatureConfig = new FeatureConfig();
  public modelConfig: OnlineModelConfig = new OnlineModelConfig();
  public decodingMethod: string = 'greedy_search';
  public maxActivePaths: number = 4;
  public enableEndpoint: boolean = false;
  public rule1MinTrailingSilence: number = 2.4;
  public rule2MinTrailingSilence: number = 1.2;
  public rule3MinUtteranceLength: number = 20;
  public hotwordsFile: string = '';
  public hotwordsScore: number = 1.5;
  public ctcFstDecoderConfig: OnlineCtcFstDecoderConfig = new OnlineCtcFstDecoderConfig();
  public ruleFsts: string = '';
  public ruleFars: string = '';
  public blankPenalty: number = 0;
}

interface OnlineRecognizerResultJson {
  text: string;
  timestamps: number[];
  tokens: string[];
}

export class OnlineRecognizerResult {
  public text: string = '';
  public tokens: string[] = [];
  public timestamps: number[] = [];
  public json: string = '';
}

export class OnlineRecognizer {
  public handle: object;
  public config: OnlineRecognizerConfig

  constructor(config: OnlineRecognizerConfig, mgr?: object) {
    this.handle = createOnlineRecognizer(config, mgr);
    this.config = config
  }

  createStream(): OnlineStream {
    const handle: object = createOnlineStream(this.handle);
    return new OnlineStream(handle);
  }

  isReady(stream: OnlineStream): boolean {
    return isOnlineStreamReady(this.handle, stream.handle);
  }

  decode(stream: OnlineStream) {
    decodeOnlineStream(this.handle, stream.handle);
  }

  isEndpoint(stream: OnlineStream): boolean {
    return isEndpoint(this.handle, stream.handle);
  }

  reset(stream: OnlineStream) {
    reset(this.handle, stream.handle);
  }

  getResult(stream: OnlineStream): OnlineRecognizerResult {
    const jsonStr: string = getOnlineStreamResultAsJson(this.handle, stream.handle);

    let o = JSON.parse(jsonStr) as OnlineRecognizerResultJson;

    const r = new OnlineRecognizerResult()
    r.text = o.text
    r.timestamps = o.timestamps;
    r.tokens = o.tokens;
    r.json = jsonStr;

    return r;
  }
}