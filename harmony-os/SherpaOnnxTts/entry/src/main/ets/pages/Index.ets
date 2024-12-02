import { CircularBuffer } from 'sherpa_onnx';
import worker, { MessageEvents } from '@ohos.worker';
import { audio } from '@kit.AudioKit';
import picker from '@ohos.file.picker';
import fs from '@ohos.file.fs';
import systemTime from '@ohos.systemTime';


function savePcmToWav(filename: string, samples: Int16Array, sampleRate: number) {
  const fp = fs.openSync(filename, fs.OpenMode.READ_WRITE | fs.OpenMode.CREATE);

  const header = new ArrayBuffer(44);
  const view = new DataView(header);

  // http://soundfile.sapp.org/doc/WaveFormat/
  //                   F F I R
  view.setUint32(0, 0x46464952, true); // chunkID
  view.setUint32(4, 36 + samples.length * 2, true); // chunkSize //                   E V A W
  view.setUint32(8, 0x45564157, true); // format //
  //                      t m f
  view.setUint32(12, 0x20746d66, true); // subchunk1ID
  view.setUint32(16, 16, true); // subchunk1Size, 16 for PCM
  view.setUint32(20, 1, true); // audioFormat, 1 for PCM
  view.setUint16(22, 1, true); // numChannels: 1 channel
  view.setUint32(24, sampleRate, true); // sampleRate
  view.setUint32(28, sampleRate * 2, true); // byteRate
  view.setUint16(32, 2, true); // blockAlign
  view.setUint16(34, 16, true); // bitsPerSample
  view.setUint32(36, 0x61746164, true); // Subchunk2ID
  view.setUint32(40, samples.length * 2, true); // subchunk2Size

  fs.writeSync(fp.fd, new Uint8Array(header).buffer, { length: header.byteLength });
  fs.writeSync(fp.fd, samples.buffer, { length: samples.buffer.byteLength });

  fs.closeSync(fp.fd);
}

function toInt16Samples(samples: Float32Array): Int16Array {
  const int16Samples = new Int16Array(samples.length);
  for (let i = 0; i < samples.length; ++i) {
    let s = samples[i] * 32767;
    s = s > 32767 ? 32767 : s;
    s = s < -32768 ? -32768 : s;
    int16Samples[i] = s;
  }

  return int16Samples;
}


@Entry
@Component
struct Index {
  @State title: string = 'Next-gen Kaldi: Text-to-speech';
  @State info: string = '';
  @State btnStartCaption: string = 'Start';
  @State btnStartEnabled: boolean = false;
  @State btnStopCaption: string = 'Stop';
  @State btnStopEnabled: boolean = false;
  @State btnSaveCaption: string = 'Save';
  @State btnSaveEnabled: boolean = false;
  @State initTtsDone: boolean = false;
  @State ttsGeneratedDone: boolean = true;
  @State sampleRate: number = 0;
  @State initAudioDone: boolean = false;
  private startTime: number = 0;
  private stopTime: number = 0;
  private inputText: string = '';
  // it specifies only the initial capacity.
  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/NonStreamingTtsWorker.ets'
  // note that circular buffer can automatically resize.
  private sampleBuffer: CircularBuffer = new CircularBuffer(16000 * 5);
  private finalSamples: Float32Array | null = null;
  private audioRenderer: audio.AudioRenderer | null = null;

  initAudioRenderer() {
    if (this.audioRenderer) {
      console.log(`Audio renderer has already been created. Skip creating`);
      return;
    } // see
    // https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/using-audiorenderer-for-playback-V5
    const audioStreamInfo: audio.AudioStreamInfo = {
      samplingRate: this.sampleRate,
      channels: audio.AudioChannel.CHANNEL_1, // 通道
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
      encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW
    };

    const audioRendererInfo: audio.AudioRendererInfo = {
      usage: audio.StreamUsage.STREAM_USAGE_MUSIC, rendererFlags: 0
    };

    const audioRendererOptions: audio.AudioRendererOptions = {
      streamInfo: audioStreamInfo, rendererInfo: audioRendererInfo
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

  async aboutToAppear() {
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

        systemTime.getRealTime((err, data) => {
          if (err) {
            console.log(`Failed to get stop time`)
          } else {
            this.stopTime = data;

            const audioDuration = samples.length / sampleRate;
            const elapsedSeconds = (this.stopTime - this.startTime) / 1000;
            const RTF = elapsedSeconds / audioDuration;
            this.info = `Audio duration: ${audioDuration} s
Elapsed: ${elapsedSeconds} s
RTF = ${elapsedSeconds.toFixed(2)}/${audioDuration.toFixed(2)} = ${RTF.toFixed(3)}
`
          }
        });

        this.finalSamples = samples;
        this.ttsGeneratedDone = true;
        this.btnSaveEnabled = true;

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
      Column({ space: 10 }) {
        Text(this.title).fontSize(20).fontWeight(FontWeight.Bold).onClick(() => {

        })

        Row({ space: 10 }) {
          Button(this.btnStartCaption).enabled(this.btnStartEnabled).onClick(async () => {
            console.log(`input text is ${this.inputText}`);

            if (this.workerInstance && this.initTtsDone) {
              this.finalSamples = null;
              this.sampleBuffer.reset();
              this.ttsGeneratedDone = false;

              this.btnStartEnabled = false;
              this.btnStopEnabled = true;
              this.btnSaveEnabled = false;
              console.log(`sending ${this.inputText}`)
              this.ttsGeneratedDone = false;
              this.startTime = await systemTime.getRealTime();
              this.workerInstance.postMessage({ msgType: 'tts-generate', text: this.inputText });
            }

          });

          Button(this.btnStopCaption).enabled(this.btnStopEnabled).onClick(() => {
            this.ttsGeneratedDone = true;
            this.btnStartEnabled = true;
            this.btnStopEnabled = false;
            this.sampleBuffer.reset();
          })

          Button(this.btnSaveCaption).enabled(this.btnSaveEnabled).onClick(() => {
            if (!this.finalSamples || this.finalSamples.length == 0) {

              this.btnSaveEnabled = false;
              return;
            }

            let uri: string = '';

            const audioOptions = new picker.AudioSaveOptions(); // audioOptions.newFileNames = ['o.wav'];

            const audioViewPicker = new picker.AudioViewPicker();

            audioViewPicker.save(audioOptions).then((audioSelectResult: Array<string>) => {
              uri = audioSelectResult[0];
              if (this.finalSamples) {
                savePcmToWav(uri, toInt16Samples(this.finalSamples), this.sampleRate);
                console.log(`Saved to ${uri}}`);
                this.info += `\nSaved to ${uri}`;
              }
            });

          })
        }

        if (this.info != '') {
          TextArea({ text: this.info });
        }

        TextArea({ placeholder: 'Input text for TTS and click the start button' })
          .width('100%')
          .height('100%')
          .onChange((text) => {
            this.inputText = text;
            if (text.trim() == '') {
              this.btnStartEnabled = false;
              return;
            }
            this.btnStartEnabled = true;
            this.info = `Audio duration: 3s;
Elapsed: 2 s
RTF: 2/3=0.66`

          })
      }.width('100%')
    }.height('100%')
  }

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
        this.btnStartEnabled = true;
        this.btnStopEnabled = false;
      }
    }
  };
}