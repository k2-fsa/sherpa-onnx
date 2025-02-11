import { LengthUnit } from '@kit.ArkUI';
import worker, { MessageEvents } from '@ohos.worker';
import { BusinessError } from '@kit.BasicServicesKit';
import { picker } from '@kit.CoreFileKit';

import { Permissions } from '@kit.AbilityKit';
import { allAllowed, requestPermissions } from './Permission';
import { audio } from '@kit.AudioKit';


@Entry
@Component
struct Index {
  @State title: string = 'Next-gen Kaldi: VAD + ASR';
  @State currentIndex: number = 0;
  @State resultForFile: string = '';
  @State progressForFile: number = 0;
  @State selectFileBtnEnabled: boolean = false;
  @State lang: string = 'English';
  @State resultForMic: string = '';
  @State micStarted: boolean = false;
  @State message: string = 'Start recording';
  @State micInitDone: boolean = false;
  private controller: TabsController = new TabsController();
  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/NonStreamingAsrWithVadWorker.ets'
  private mic?: audio.AudioCapturer;
  private sampleList: Float32Array[] = []

  flatten(samples: Float32Array[]): Float32Array {
    let n = 0;
    for (let i = 0; i < samples.length; ++i) {
      n += samples[i].length;
    }

    const ans: Float32Array = new Float32Array(n);
    let offset: number = 0;
    for (let i = 0; i < samples.length; ++i) {
      ans.set(samples[i], offset);
      offset += samples[i].length;
    }

    return ans;
  }

  async initMic() {
    const permissions: Permissions[] = ["ohos.permission.MICROPHONE"];
    let allowed: boolean = await allAllowed(permissions);
    if (!allowed) {
      console.log("request to access the microphone");
      const status: boolean = await requestPermissions(permissions);

      if (!status) {
        console.error('access to microphone is denied')
        this.resultForMic = "Failed to get microphone permission. Please retry";
        return;
      }

      allowed = await allAllowed(permissions);
      if (!allowed) {
        console.error('failed to get microphone permission');
        this.resultForMic = "Failed to get microphone permission. Please retry";
        return;
      }
    } else {
      console.log("allowed to access microphone");
    }

    const audioStreamInfo: audio.AudioStreamInfo = {
      samplingRate: audio.AudioSamplingRate.SAMPLE_RATE_16000,
      channels: audio.AudioChannel.CHANNEL_1,
      sampleFormat: audio.AudioSampleFormat.SAMPLE_FORMAT_S16LE,
      encodingType: audio.AudioEncodingType.ENCODING_TYPE_RAW,
    };

    const audioCapturerInfo: audio.AudioCapturerInfo = {
      source: audio.SourceType.SOURCE_TYPE_MIC, capturerFlags: 0
    };

    const audioCapturerOptions: audio.AudioCapturerOptions = {
      streamInfo: audioStreamInfo, capturerInfo: audioCapturerInfo

    };
    audio.createAudioCapturer(audioCapturerOptions, (err, data) => {
      if (err) {
        console.error(`error code is ${err.code}, error message is ${err.message}`);
        this.resultForMic = 'Failed to init microphone';
      } else {
        console.info(`init mic successfully`);
        this.mic = data;
        this.mic.on('readData', this.micCallback);

        if (this.workerInstance) {
          this.workerInstance.postMessage({ msgType: 'init-vad-mic', context: getContext() });
        }
      }
    });
  }

  async aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'NonStreaming ASR worker'
    });

    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'init-vad-mic-done') {
        this.micInitDone = true;
      }

      if (msgType == 'init-non-streaming-asr-done') {
        this.selectFileBtnEnabled = true;
        this.resultForFile = `Initializing done.\n\nPlease select a wave file of 16kHz in language ${this.lang}`;
      }

      if (msgType == 'non-streaming-asr-vad-decode-done') {
        this.resultForFile = e.data['text'] as string + '\n';
      }

      if (msgType == 'non-streaming-asr-vad-decode-partial') {
        if (this.resultForFile == '') {
          this.resultForFile = e.data['text'] as string;
        } else {
          this.resultForFile += '\n\n' + e.data['text'] as string;
        }
      }

      if (msgType == 'non-streaming-asr-vad-decode-error') {
        this.resultForFile = e.data['text'] as string;
      }

      if (msgType == 'non-streaming-asr-vad-decode-progress') {
        this.progressForFile = e.data['progress'] as number;

        this.selectFileBtnEnabled = this.progressForFile >= 100;
      }

      if (msgType == 'non-streaming-asr-vad-mic-partial') {
        if (this.resultForMic == '') {
          this.resultForMic = e.data['text'] as string;
        } else {
          this.resultForMic += '\n\n' + e.data['text'] as string;
        }
      }

      if (msgType == 'non-streaming-asr-vad-mic-error') {
        this.resultForMic = e.data['text'] as string;
      }
    }

    const context = getContext();
    this.resultForFile = 'Initializing models';
    this.workerInstance.postMessage({ msgType: 'init-vad', context });
    this.workerInstance.postMessage({ msgType: 'init-non-streaming-asr', context });

    await this.initMic();
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

            Button('Select .wav file (16kHz) ')
              .enabled(this.selectFileBtnEnabled)
              .fontSize(13)
              .width(296)
              .height(60)
              .onClick(() => {
                this.resultForFile = '';
                this.progressForFile = 0;

                const documentSelectOptions = new picker.DocumentSelectOptions();
                documentSelectOptions.maxSelectNumber = 1;
                documentSelectOptions.fileSuffixFilters = ['.wav'];
                const documentViewPicker = new picker.DocumentViewPicker();
                documentViewPicker.select(documentSelectOptions).then((result: Array<string>) => {
                  console.log(`Result: ${result}`);

                  if (!result[0]) {
                    this.resultForFile = 'Please select a file to decode';
                    this.selectFileBtnEnabled = true;
                    return;
                  }

                  if (this.workerInstance) {
                    this.workerInstance.postMessage({
                      msgType: 'non-streaming-asr-vad-decode', filename: result[0],
                    });
                  } else {
                    console.log(`this worker instance is undefined ${this.workerInstance}`);
                  }
                }).catch((err: BusinessError) => {
                  console.error(`Failed to select file, code is ${err.code}, message is ${err.message}`);
                })

              })

            Text(`Supported languages: ${this.lang}`)

            if (this.progressForFile > 0) {
              Row() {
                Progress({ value: 0, total: 100, type: ProgressType.Capsule })
                  .width('80%')
                  .height(20)
                  .value(this.progressForFile);

                Text(`${this.progressForFile.toFixed(2)}%`).width('15%')
              }.width('100%').justifyContent(FlexAlign.Center)
            }

            TextArea({ text: this.resultForFile })
              .width('100%')
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .height('100%');
          }.alignItems(HorizontalAlign.Center).justifyContent(FlexAlign.Start)
        }.tabBar(this.TabBuilder('From file', 0, $r('app.media.icon_doc'), $r('app.media.icon_doc')))

        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(20).fontWeight(FontWeight.Bold);
            Button(this.message).enabled(this.micInitDone).onClick(() => {
              console.log('clicked mic button');
              this.resultForMic = '';
              if (this.mic) {
                if (this.micStarted) {
                  this.mic.stop();
                  this.message = "Start recording";
                  this.micStarted = false;
                  console.log('mic stopped');

                  const samples = this.flatten(this.sampleList);
                  let s = 0;
                  for (let i = 0; i < samples.length; ++i) {
                    s += samples[i];
                  }
                  console.log(`samples ${samples.length}, sum: ${s}`);

                  if (this.workerInstance) {
                    console.log('decode mic');
                    this.workerInstance.postMessage({
                      msgType: 'non-streaming-asr-vad-mic', samples,
                    });
                  } else {
                    console.log(`this worker instance is undefined ${this.workerInstance}`);
                  }
                } else {
                  this.sampleList = [];
                  this.mic.start();
                  this.message = "Stop recording";
                  this.micStarted = true;
                  console.log('mic started');
                }
              }
            });

            Text(`Supported languages: ${this.lang}`)

            TextArea({ text: this.resultForMic })
              .width('100%')
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .width('100%')
              .height('100%');
          }.alignItems(HorizontalAlign.Center).justifyContent(FlexAlign.Start)
        }
        .tabBar(this.TabBuilder('From mic', 1, $r('app.media.icon_mic'),
          $r('app.media.icon_mic')))

        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(20).fontWeight(FontWeight.Bold);
            TextArea({
              text: `
Everyting is open-sourced.

It runs locally, without accessing the network

See also https://github.com/k2-fsa/sherpa-onnx

新一代 Kaldi QQ 和微信交流群: 请看

https://k2-fsa.github.io/sherpa/social-groups.html

微信公众号: 新一代 Kaldi
            `
            }).width('100%').height('100%').focusable(false)
          }.justifyContent(FlexAlign.Start)
        }.tabBar(this.TabBuilder('Help', 2, $r('app.media.info'), $r('app.media.info')))

      }.scrollable(false)
    }.width('100%').justifyContent(FlexAlign.Start)
  }

  private micCallback = (buffer: ArrayBuffer) => {
    const view: Int16Array = new Int16Array(buffer);

    const samplesFloat: Float32Array = new Float32Array(view.length);
    for (let i = 0; i < view.length; ++i) {
      samplesFloat[i] = view[i] / 32768.0;
    }
    this.sampleList.push(samplesFloat);
  }
}