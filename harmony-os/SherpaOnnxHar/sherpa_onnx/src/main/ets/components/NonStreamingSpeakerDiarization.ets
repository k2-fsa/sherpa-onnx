import {
  createOfflineSpeakerDiarization,
  getOfflineSpeakerDiarizationSampleRate,
  offlineSpeakerDiarizationProcess,
  offlineSpeakerDiarizationProcessAsync,
  offlineSpeakerDiarizationSetConfig,
} from 'libsherpa_onnx.so';

import { SpeakerEmbeddingExtractorConfig } from './SpeakerIdentification';

export class OfflineSpeakerSegmentationPyannoteModelConfig {
  public model: string = '';
}

export class OfflineSpeakerSegmentationModelConfig {
  public pyannote: OfflineSpeakerSegmentationPyannoteModelConfig = new OfflineSpeakerSegmentationPyannoteModelConfig();
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class FastClusteringConfig {
  public numClusters: number = -1;
  public threshold: number = 0.5;
}

export class OfflineSpeakerDiarizationConfig {
  public segmentation: OfflineSpeakerSegmentationModelConfig = new OfflineSpeakerSegmentationModelConfig();
  public embedding: SpeakerEmbeddingExtractorConfig = new SpeakerEmbeddingExtractorConfig();
  public clustering: FastClusteringConfig = new FastClusteringConfig();
  public minDurationOn: number = 0.2;
  public minDurationOff: number = 0.5;
}

export class OfflineSpeakerDiarizationSegment {
  // in seconds
  public start: number = 0;
  // in seconds
  public end: number = 0;
  // ID of the speaker; count from 0
  public speaker: number = 0;
}

export class OfflineSpeakerDiarization {
  public config: OfflineSpeakerDiarizationConfig;
  public sampleRate: number;
  private handle: object;

  constructor(config: OfflineSpeakerDiarizationConfig, mgr?: object) {
    this.handle = createOfflineSpeakerDiarization(config, mgr);
    this.config = config;

    this.sampleRate = getOfflineSpeakerDiarizationSampleRate(this.handle);
  }

  /**
   * samples is a 1-d float32 array. Each element of the array should be
   * in the range [-1, 1].
   *
   * We assume its sample rate equals to this.sampleRate.
   *
   * Returns an array of object, where an object is
   *
   *  {
   *    "start": start_time_in_seconds,
   *    "end": end_time_in_seconds,
   *    "speaker": an_integer,
   *  }
   */
  process(samples: Float32Array): OfflineSpeakerDiarizationSegment[] {
    return offlineSpeakerDiarizationProcess(this.handle, samples) as OfflineSpeakerDiarizationSegment[];
  }

  processAsync(samples: Float32Array, callback: (numProcessedChunks: number,
    numTotalChunks: number) => void): Promise<OfflineSpeakerDiarizationSegment[]> {
    return offlineSpeakerDiarizationProcessAsync(this.handle, samples,
      callback) as Promise<OfflineSpeakerDiarizationSegment[]>;
  }

  setConfig(config: OfflineSpeakerDiarizationConfig) {
    offlineSpeakerDiarizationSetConfig(this.handle, config);
    this.config.clustering = config.clustering;
  }
}
