import {
  createOnlinePunctuation,
  onlinePunctuationAddPunct,
} from 'libsherpa_onnx.so';

export class OnlinePunctuationModelConfig {
  public cnnBilstm: string = '';
  public bpeVocab: string = '';
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OnlinePunctuationConfig {
  public model: OnlinePunctuationModelConfig = new OnlinePunctuationModelConfig();
}

export class OnlinePunctuation {
  public config: OnlinePunctuationConfig;
  private handle: object;

  constructor(config: OnlinePunctuationConfig, mgr?: object) {
    this.handle = createOnlinePunctuation(config, mgr);
    this.config = config;
  }

  addPunct(text: string): string {
    return onlinePunctuationAddPunct(this.handle, text);
  }
}
