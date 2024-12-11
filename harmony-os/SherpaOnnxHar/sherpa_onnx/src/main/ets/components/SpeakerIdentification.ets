import {
  createSpeakerEmbeddingExtractor,
  createSpeakerEmbeddingManager,
  speakerEmbeddingExtractorComputeEmbedding,
  speakerEmbeddingExtractorCreateStream,
  speakerEmbeddingExtractorDim,
  speakerEmbeddingExtractorIsReady,
  speakerEmbeddingManagerAdd,
  speakerEmbeddingManagerAddListFlattened,
  speakerEmbeddingManagerContains,
  speakerEmbeddingManagerGetAllSpeakers,
  speakerEmbeddingManagerNumSpeakers,
  speakerEmbeddingManagerRemove,
  speakerEmbeddingManagerSearch,
  speakerEmbeddingManagerVerify
} from 'libsherpa_onnx.so';
import { OnlineStream } from './StreamingAsr';

export class SpeakerEmbeddingExtractorConfig {
  public model: string = '';
  public numThreads: number = 1;
  public debug: boolean = false;
  public provider: string = 'cpu';
}

export class SpeakerEmbeddingExtractor {
  public config: SpeakerEmbeddingExtractorConfig = new SpeakerEmbeddingExtractorConfig();
  public dim: number;
  private handle: object;

  constructor(config: SpeakerEmbeddingExtractorConfig, mgr?: object) {
    this.handle = createSpeakerEmbeddingExtractor(config, mgr);
    this.config = config;
    this.dim = speakerEmbeddingExtractorDim(this.handle);
  }

  createStream(): OnlineStream {
    return new OnlineStream(speakerEmbeddingExtractorCreateStream(this.handle));
  }

  isReady(stream: OnlineStream): boolean {
    return speakerEmbeddingExtractorIsReady(this.handle, stream.handle);
  }

  compute(stream: OnlineStream, enableExternalBuffer: boolean = true): Float32Array {
    return speakerEmbeddingExtractorComputeEmbedding(this.handle, stream.handle, enableExternalBuffer);
  }
}

function flatten(arrayList: Float32Array[]): Float32Array {
  let n = 0;
  for (let i = 0; i < arrayList.length; ++i) {
    n += arrayList[i].length;
  }
  let ans = new Float32Array(n);

  let offset = 0;
  for (let i = 0; i < arrayList.length; ++i) {
    ans.set(arrayList[i], offset);
    offset += arrayList[i].length;
  }
  return ans;
}

interface SpeakerNameWithEmbedding {
  name: string;
  v: Float32Array;
}

interface SpeakerNameWithEmbeddingList {
  name: string;
  v: Float32Array[];
}

interface SpeakerNameWithEmbeddingN {
  name: string;
  vv: Float32Array;
  n: number;
}

interface EmbeddingWithThreshold {
  v: Float32Array;
  threshold: number;
}

interface SpeakerNameEmbeddingThreshold {
  name: string;
  v: Float32Array;
  threshold: number;
}

export class SpeakerEmbeddingManager {
  public dim: number;
  private handle: object;

  constructor(dim: number) {
    this.handle = createSpeakerEmbeddingManager(dim);
    this.dim = dim;
  }

  add(speaker: SpeakerNameWithEmbedding): boolean {
    return speakerEmbeddingManagerAdd(this.handle, speaker);
  }

  addMulti(speaker: SpeakerNameWithEmbeddingList): boolean {
    const c: SpeakerNameWithEmbeddingN = {
      name: speaker.name, vv: flatten(speaker.v), n: speaker.v.length,
    };
    return speakerEmbeddingManagerAddListFlattened(this.handle, c);
  }

  remove(name: string): boolean {
    return speakerEmbeddingManagerRemove(this.handle, name);
  }

  search(obj: EmbeddingWithThreshold): string {
    return speakerEmbeddingManagerSearch(this.handle, obj);
  }

  verify(obj: SpeakerNameEmbeddingThreshold): boolean {
    return speakerEmbeddingManagerVerify(this.handle, obj);
  }

  contains(name: string): boolean {
    return speakerEmbeddingManagerContains(this.handle, name);
  }

  getNumSpeakers(): number {
    return speakerEmbeddingManagerNumSpeakers(this.handle);
  }

  getAllSpeakerNames(): string[] {
    return speakerEmbeddingManagerGetAllSpeakers(this.handle);
  }
}