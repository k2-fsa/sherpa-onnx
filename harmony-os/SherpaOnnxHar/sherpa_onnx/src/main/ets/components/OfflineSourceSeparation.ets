import {
  createOfflineSourceSeparation,
  offlineSourceSeparationGetNumberOfStems,
  offlineSourceSeparationGetOutputSampleRate,
  offlineSourceSeparationProcess,
} from 'libsherpa_onnx.so';

export class OfflineSourceSeparationSpleeterModelConfig {
  public vocals: string = '';
  public accompaniment: string = '';
}

export class OfflineSourceSeparationUvrModelConfig {
  public model: string = '';
}

export class OfflineSourceSeparationModelConfig {
  public spleeter: OfflineSourceSeparationSpleeterModelConfig = new OfflineSourceSeparationSpleeterModelConfig();
  public uvr: OfflineSourceSeparationUvrModelConfig = new OfflineSourceSeparationUvrModelConfig();
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class OfflineSourceSeparationConfig {
  public model: OfflineSourceSeparationModelConfig = new OfflineSourceSeparationModelConfig();
}

export class SourceSeparationInput {
  /**
   * One or two channels of equal, nonzero length. Each element of a channel
   * should be in the range [-1, 1]. Mono input is duplicated to stereo
   * internally by sherpa-onnx; you don't need to duplicate it yourself.
   */
  public samples: Float32Array[] = [];
  public sampleRate: number = 0;
  public enableExternalBuffer: boolean = true;
}

export class SourceSeparationOutput {
  /**
   * stems[s][c] is channel c of stem s. For the models above, stems[0] is
   * vocals and stems[1] is accompaniment (non-vocals).
   */
  public stems: Float32Array[][] = [];
  public sampleRate: number = 0;
}

export class OfflineSourceSeparation {
  public config: OfflineSourceSeparationConfig;
  // Output sample rate of the model. Input is resampled to it if needed.
  public sampleRate: number;
  public numStems: number;
  private handle: object;

  constructor(config: OfflineSourceSeparationConfig, mgr?: object) {
    this.handle = createOfflineSourceSeparation(config, mgr);
    this.config = config;

    this.sampleRate = offlineSourceSeparationGetOutputSampleRate(this.handle);
    this.numStems = offlineSourceSeparationGetNumberOfStems(this.handle);
  }

  /**
   * Run source separation synchronously.
   *
   * Returns an object {stems, sampleRate}, where stems is an array of stems
   * and each stem is an array of Float32Array channels.
   */
  process(input: SourceSeparationInput): SourceSeparationOutput {
    return offlineSourceSeparationProcess(this.handle, input);
  }
}
