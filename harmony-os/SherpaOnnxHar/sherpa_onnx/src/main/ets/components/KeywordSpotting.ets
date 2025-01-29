import {
  createKeywordSpotter,
  createKeywordStream,
  isKeywordStreamReady,
  decodeKeywordStream,
  resetKeywordStream,
  getKeywordResultAsJson,
} from 'libsherpa_onnx.so';

import { FeatureConfig } from './NonStreamingAsr';
import { OnlineModelConfig, OnlineStream } from './StreamingAsr';

export class KeywordSpotterConfig {
  public featConfig: FeatureConfig = new FeatureConfig();
  public modelConfig: OnlineModelConfig = new OnlineModelConfig();
  public maxActivePaths: number = 4;
  public numTrailingBlanks: number = 1;
  public keywordsScore: number = 1;
  public keywordsThreshold: number = 0.25;
  public keywordsFile: string = '';
}

interface KeywordSpotterResultJson {
  keyword: string;
  timestamps: number[];
  tokens: string[];
}

export class KeywordSpotterResult {
  public keyword: string = '';
  public tokens: string[] = [];
  public timestamps: number[] = [];
  public json: string = '';
}

export class KeywordSpotter {
  public handle: object;
  public config: KeywordSpotterConfig;

  constructor(config: KeywordSpotterConfig, mgr?: object) {
    this.handle = createKeywordSpotter(config, mgr);
    this.config = config
  }

  createStream(keywords?: string): OnlineStream {
    if (typeof keywords !== "undefined") {
      return new OnlineStream(createKeywordStream(this.handle, keywords));
    } else {
      return new OnlineStream(createKeywordStream(this.handle));
    }
  }

  isReady(stream: OnlineStream): boolean {
    return isKeywordStreamReady(this.handle, stream.handle);
  }

  decode(stream: OnlineStream) {
    decodeKeywordStream(this.handle, stream.handle);
  }

  reset(stream: OnlineStream) {
    resetKeywordStream(this.handle, stream.handle);
  }

  getResult(stream: OnlineStream): KeywordSpotterResult {
    const jsonStr: string = getKeywordResultAsJson(this.handle, stream.handle);

    let o = JSON.parse(jsonStr) as KeywordSpotterResultJson;

    const r = new KeywordSpotterResult()
    r.keyword = o.keyword
    r.timestamps = o.timestamps;
    r.tokens = o.tokens;
    r.json = jsonStr;

    return r;
  }
}
