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
  view.setUint32(8, 0x45564157, true); // format // //                      t m f
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
  @State currentIndex: number = 0;
  @State title: string = 'Next-gen Kaldi: Text-to-speech';
  @State info: string = '';
  @State btnStartCaption: string = 'Start';
  @State btnStartEnabled: boolean = false;
  @State btnStopCaption: string = 'Stop';
  @State btnStopEnabled: boolean = false;
  @State btnSaveCaption: string = 'Save';
  @State btnSaveEnabled: boolean = false;
  @State progress: number = 0;
  @State sid: string = '0';
  @State speechSpeed: string = '1.0';
  @State isGenerating: boolean = false;
  @State initTtsDone: boolean = false;
  @State ttsGeneratedDone: boolean = true;
  @State numSpeakers: number = 1;
  @State initAudioDone: boolean = false;
  private controller: TabsController = new TabsController();
  private cancelled: boolean = false;
  private sampleRate: number = 0;
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
    } // see // https://developer.huawei.com/consumer/cn/doc/harmonyos-guides-V5/using-audiorenderer-for-playback-V5
    console.log('Initializing audio renderer');
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
      } else {
        console.log(`Failed to initialize audio renderer. error message: ${err.message}, error code: ${err.code}`);
      }
    });
  }

  async aboutToAppear() {
    this.initAudioRenderer();

    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'NonStreaming TTS worker'
    });
    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'init-tts-done') {
        this.info = 'Model initialized!\nPlease enter text and press start.';
        this.sampleRate = e.data['sampleRate'] as number;
        this.numSpeakers = e.data['numSpeakers'] as number;

        this.initTtsDone = true;
      }

      if (msgType == 'tts-generate-partial') {
        if (this.cancelled) {
          return;
        }

        const samples: Float32Array = e.data['samples'] as Float32Array;
        const progress: number = e.data['progress'] as number;
        this.progress = progress;

        this.sampleBuffer.push(samples);

        if (!this.initAudioDone) {
          this.initAudioRenderer();
        }

        if (this.audioRenderer && this.audioRenderer?.state != audio.AudioState.STATE_RUNNING) {
          this.audioRenderer.start();
        }
      }

      if (msgType == 'tts-generate-done') {
        this.isGenerating = false;
        const samples: Float32Array = e.data['samples'] as Float32Array;

        systemTime.getRealTime((err, data) => {

          if (err) {
            console.log(`Failed to get stop time`)
          } else {
            this.stopTime = data;

            const audioDuration = samples.length / this.sampleRate;
            const elapsedSeconds = (this.stopTime - this.startTime) / 1000;
            const RTF = elapsedSeconds / audioDuration;

            this.info = `Audio duration: ${audioDuration} s
Elapsed: ${elapsedSeconds} s
RTF = ${elapsedSeconds.toFixed(2)}/${audioDuration.toFixed(2)} = ${RTF.toFixed(3)}
`;
            if (this.cancelled) {
              this.info += '\nCancelled.';
            }
          }
        });

        this.finalSamples = samples;
        this.ttsGeneratedDone = true;
        this.btnSaveEnabled = true;

        this.ttsGeneratedDone = true;

        if (this.audioRenderer && this.audioRenderer?.state != audio.AudioState.STATE_RUNNING &&
          this.sampleBuffer.size() == 0) {
          this.sampleBuffer.push(samples);
          this.progress = 1;
          this.audioRenderer.start();
        }

        if (!this.initAudioDone) {
          this.btnStartEnabled = true;
          this.btnStopEnabled = false;
          this.info += '\nAudio renderer is not initialized. Disable playing audio.';
        }
      }
    }

    this.info = 'Initializing TTS model ...';
    this.workerInstance.postMessage({ msgType: 'init-tts', context: getContext() });
  }

  @Builder
  TabBuilder(title: string, targetIndex: number, selectedImg: Resource, normalImg: Resource) {
    Column() {
      Image(this.currentIndex == targetIndex ? selectedImg : normalImg).size({ width: 25, height: 25 })
      Text(title).fontColor(this.currentIndex == targetIndex ? '#28bff1' : '#8a8a8a')
    }.width('100%').height(50).justifyContent(FlexAlign.Center).onClick(() => {
      this.currentIndex = targetIndex;
      this.controller.changeIndex(this.currentIndex);
    })
  }

  build() {
    Column() {
      Tabs({ barPosition: BarPosition.End, controller: this.controller }) {
        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(20).fontWeight(FontWeight.Bold);
            if (this.numSpeakers > 1) {
              Row({ space: 10 }) {
                Text(`Speaker ID (0-${this.numSpeakers - 1})`).width('60%')

                TextInput({ text: this.sid }).onChange((text) => {
                  this.sid = text.trim();
                }).width('20%')
              }.justifyContent(FlexAlign.Center)
            }

            Row() {
              Text('Speech speed').width('60%');

              TextInput({ text: this.speechSpeed }).onChange((text) => {
                this.speechSpeed = text.trim();
              }).width('20%')
            }

            Row({ space: 10 }) {
              Button(this.btnStartCaption).enabled(this.btnStartEnabled).onClick(async () => {
                let sid = parseInt(this.sid);
                if (sid.toString() != this.sid) {
                  this.info = 'Please input a valid speaker ID';
                  return;
                }

                let speed = parseFloat(this.speechSpeed);
                if (isNaN(speed)) {
                  this.info = 'Please enter a valid speech speed';
                  return;
                }

                if (speed <= 0) {
                  this.info = 'Please enter a positive speech speed';
                  return;
                }

                if (this.workerInstance && this.initTtsDone) {
                  this.info = 'Generating...';
                  this.cancelled = false;
                  this.finalSamples = null;
                  this.sampleBuffer.reset();
                  this.ttsGeneratedDone = false;
                  this.progress = 0;

                  this.btnStartEnabled = false;
                  this.btnStopEnabled = true;
                  this.btnSaveEnabled = false;
                  console.log(`sending ${this.inputText}`)
                  this.ttsGeneratedDone = false;
                  this.startTime = await systemTime.getRealTime();
                  this.workerInstance?.postMessage({
                    msgType: 'tts-generate',
                    text: this.inputText,
                    sid: sid,
                    speed: speed,
                  });
                  this.isGenerating = true;
                  this.info = '';
                } else {
                  this.info = 'Failed to initialize tts model';
                  this.btnStartEnabled = false;
                }
              });

              Button(this.btnStopCaption).enabled(this.btnStopEnabled).onClick(() => {
                this.ttsGeneratedDone = true;
                this.btnStartEnabled = true;
                this.btnStopEnabled = false;
                this.sampleBuffer.reset();
                this.cancelled = true;
                this.isGenerating = false;

                if (this.workerInstance && this.initTtsDone) {
                  this.workerInstance.postMessage({ msgType: 'tts-generate-cancel' });
                }
                this.audioRenderer?.stop();
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
                    console.log(`Saved to ${uri}`);
                    this.info += `\nSaved to ${uri}`;
                  }
                });
              });
            }

            if (this.info != '') {
              TextArea({ text: this.info }).focusable(false);
            }
            if (this.progress > 0) {
              Row() {
                Progress({ value: 0, total: 100, type: ProgressType.Capsule })
                  .width('80%')
                  .height(20)
                  .value(this.progress * 100);

                Text(`${(this.progress * 100).toFixed(2)}%`).width('15%')
              }.width('100%').justifyContent(FlexAlign.Center)
            }

            TextArea({ placeholder: 'Input text for TTS and click the start button' })
              .width('100%')
              .height('100%')
              .focusable(this.isGenerating == false && this.initTtsDone)
              .onChange((text) => {
                this.inputText = text;
                if (text.trim() == '') {
                  this.btnStartEnabled = false;
                  return;
                }
                this.btnStartEnabled = true;
              })
          }.width('100%')

          // see https://composeicons.com/
        }.tabBar(this.TabBuilder('TTS', 0, $r('app.media.home'), $r('app.media.home')))

        TabContent() {
          Column({space: 10}) {
            Text(this.title).fontSize(20).fontWeight(FontWeight.Bold);
            TextArea({text: `
Everyting is open-sourced.

It runs locally, without accessing the network

See also https://github.com/k2-fsa/sherpa-onnx

新一代 Kaldi QQ 和微信交流群: 请看

https://k2-fsa.github.io/sherpa/social-groups.html

微信公众号: 新一代 Kaldi
            `}).width('100%')
              .height('100%')
              .focusable(false)
          }.justifyContent(FlexAlign.Start)
        }.tabBar(this.TabBuilder('Help', 1, $r('app.media.info'), $r('app.media.info')))
      }.scrollable(false)
    }
  }

  private audioPlayCallback = (buffer: ArrayBuffer) => {
    const numSamples = buffer.byteLength / 2;
    if (this.sampleBuffer.size() >= numSamples) {
      const samples: Float32Array = this.sampleBuffer.get(this.sampleBuffer.head(), numSamples);

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