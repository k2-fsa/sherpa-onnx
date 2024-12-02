import { CircularBuffer } from 'sherpa_onnx';
import worker, { MessageEvents } from '@ohos.worker';
import { audio } from '@kit.AudioKit';



@Entry
@Component
struct Index {
  @State message: string = 'Hello World';
  @State initTtsDone: boolean = false;
  @State ttsGeneratedDone: boolean = true;

  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/NonStreamingTtsWorker.ets'

  // it specifies only the initial capacity.
  // note that circular buffer can automatically resize.
  private sampleBuffer: CircularBuffer = new CircularBuffer(16000 * 5);
  @State sampleRate: number = 0;

  private audioRenderer: audio.AudioRenderer | null = null;
  @State initAudioDone: boolean = false;

  private audioPlayCallback = (buffer: ArrayBuffer) => {
    const numSamples = buffer.byteLength / 2;
    if (this.sampleBuffer.size() >= numSamples) {
      const samples = this.sampleBuffer.get(this.sampleBuffer.head(), numSamples);
      const int16Samples = new Int16Array(buffer);
      for (let i = 0; i < numSamples; ++i) {
        let s = samples[i] * 32767;
        s = s > 32767 ? 32767 : s;
        s = s < -32768 ? -32768 : s;
        int16Samples[i] = s;
      }
      this.sampleBuffer.pop(numSamples);
    } else {
      (new Int16Array(buffer)).fill(0);
      if (this.ttsGeneratedDone) {
        this.audioRenderer?.stop();
      }
    }
  };

  initAudioRenderer() {
    if (this.audioRenderer) {
      console.log(`Audio renderer has already been created. Skip creating`);
      return;
    }
    // see
    // https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/using-audiorenderer-for-playback-V5
    const audioStreamInfo: audio.AudioStreamInfo = {
      samplingRate: this.sampleRate,
      channels: audio.AudioChannel.CHANNEL_1, // 通道
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
      encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
    };

    const audioRendererInfo: audio.AudioRendererInfo = {
      usage: audio.StreamUsage.STREAM_USAGE_MUSIC,
      rendererFlags: 0
    };

    const audioRendererOptions: audio.AudioRendererOptions = {
      streamInfo: audioStreamInfo,
      rendererInfo: audioRendererInfo
    };

    audio.createAudioRenderer(audioRendererOptions, (err, renderer) => {
      if (!err) {
        console.log('audio renderer initialized successfully');
        this.initAudioDone = true;
        if (renderer) {
          this.audioRenderer = renderer;
          this.audioRenderer.on("writeData", this.audioPlayCallback);
          if (this.sampleBuffer.size()) {
            this.audioRenderer.start();
          }
        } else {
          console.log(`returned audio renderer is ${renderer}`);
        }
      }
    });
  }

  aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'NonStreaming TTS worker'
    });
    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'init-tts-done') {
        console.log('init tts done');
        this.initTtsDone = true;
      }

      if (msgType == 'tts-generate-done') {
        const samples: Float32Array = e.data['samples'] as Float32Array;
        const sampleRate = e.data['sampleRate'] as number;
        if (this.sampleRate == 0) {
          console.log(`sample rate is ${sampleRate}`);
          this.sampleRate = sampleRate;
        }
        this.sampleBuffer.push(samples);
        this.ttsGeneratedDone = true;
        if (!this.initAudioDone) {
          this.initAudioRenderer();
        }

        if (this.audioRenderer && this.audioRenderer?.state != audio.AudioState.STATE_RUNNING) {
          this.audioRenderer.start();
        }
      }
    }

    this.workerInstance.postMessage({ msgType: 'init-tts', context: getContext() });
  }

  build() {
    Row() {
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(()=>{
            if (this.workerInstance && this.initTtsDone) {
              const text = 'how are you doing?';
              console.log(`sending ${text}`)
              this.ttsGeneratedDone = false;
              this.workerInstance.postMessage({msgType: 'tts-generate', text});
            }

          })
      }
      .width('100%')
    }
    .height('100%')
  }
}