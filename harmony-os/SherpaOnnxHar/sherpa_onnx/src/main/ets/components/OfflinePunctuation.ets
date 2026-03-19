import {
  createOfflinePunctuation,
  offlinePunctuationAddPunct,
} from 'libsherpa_onnx.so';

export class OfflinePunctuationModelConfig {
  public ctTransformer: string = '';
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OfflinePunctuationConfig {
  public model: OfflinePunctuationModelConfig = new OfflinePunctuationModelConfig();
}

export class OfflinePunctuation {
  public config: OfflinePunctuationConfig;
  private handle: object;

  constructor(config: OfflinePunctuationConfig, mgr?: object) {
    this.handle = createOfflinePunctuation(config, mgr);
    this.config = config;
  }

  addPunct(text: string): string {
    return offlinePunctuationAddPunct(this.handle, text);
  }
}
