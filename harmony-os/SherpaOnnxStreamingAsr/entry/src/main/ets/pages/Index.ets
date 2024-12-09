import { LengthUnit } from '@kit.ArkUI';
import worker, { MessageEvents } from '@ohos.worker';
import { BusinessError } from '@kit.BasicServicesKit';
import { picker } from '@kit.CoreFileKit';
import systemTime from '@ohos.systemTime';
import { Permissions } from '@kit.AbilityKit';
import { allAllowed, requestPermissions } from './Permission';
import { audio } from '@kit.AudioKit';
import fs from '@ohos.file.fs';


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
  @State title: string = 'Next-gen Kaldi: Real-time speech recognition';
  @State titleFontSize: number = 15;
  @State currentIndex: number = 0;
  @State lang: string = 'English';
  @State resultForFile: string = ''
  @State resultForMic: string = ''
  @State selectFileBtnEnabled: boolean = false;
  @State micBtnCaption: string = 'Start';
  @State micStarted: boolean = false;
  @State micAllowed: boolean = false;
  @State micBtnEnabled: boolean = false;
  @State micSaveBtnCaption: string = 'Save recorded audio';
  @State micSaveBtnEnabled: boolean = false;
  @State info: string = '';
  @State micInfo: string = '';
  @State micInitDone: boolean = false;
  private resultListForMic: string[] = [];
  private controller: TabsController = new TabsController();
  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/StreamingAsrWorker.ets'
  private startTime: number = 0;
  private stopTime: number = 0;
  private sampleRate: number = 48000;
  private sampleList: Float32Array[] = []
  private mic?: audio.AudioCapturer;

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
      this.micAllowed = true;
    } else {
      console.log("allowed to access microphone");
      this.micAllowed = true;
    }

    const audioStreamInfo: audio.AudioStreamInfo = {
      samplingRate: this.sampleRate,
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
      }
    });
  }

  async aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'Streaming ASR worker'
    });

    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'init-streaming-asr-done') {
        this.selectFileBtnEnabled = true;
        this.micBtnEnabled = true;
        this.info = `Initializing done.\n\nPlease select a wave file of 16kHz in language ${this.lang}`;
        this.micInfo = `Initializing done.\n\nPlease click Start and speak`;
      }

      if (msgType == 'streaming-asr-decode-file-done') {
        const text = e.data['text'] as string;
        this.resultForFile = text;
        this.selectFileBtnEnabled = true;

        systemTime.getRealTime((err, data) => {
          if (err) {
            console.log('Failed to get stop time');
          } else {
            this.stopTime = data;

            const audioDuration = e.data['duration'] as number;
            const elapsedSeconds = (this.stopTime - this.startTime) / 1000;
            const RTF = elapsedSeconds / audioDuration;
            this.info = `Audio duration: ${audioDuration.toFixed(2)} s
Elapsed: ${elapsedSeconds.toFixed(2)} s
RTF = ${elapsedSeconds.toFixed(2)}/${audioDuration.toFixed(2)} = ${RTF.toFixed(3)}
`;
          }
        });
      }

      if (msgType == 'streaming-asr-decode-mic-result') {
        const text = e.data['text'] as string;
        if (text.trim() == '') {
          return;
        }

        const isEndpoint = e.data['isEndpoint'] as boolean;

        let s = '';
        let i = 0;
        for (; i < this.resultListForMic.length; ++i) {
          s += `${i}: ${this.resultListForMic[i]}\n`
        }

        s += `${i}: ${text}`;
        this.resultForMic = s;

        if (isEndpoint) {
          this.resultListForMic.push(text);
        }
      }
    };

    const context = getContext();
    this.workerInstance.postMessage({ msgType: 'init-streaming-asr', context });
    this.info = 'Initializing ASR model.\nPlease wait';
    this.micInfo = 'Initializing ASR model.\nPlease wait';

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
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
            Button('Select .wav file (16kHz) ')
              .enabled(this.selectFileBtnEnabled)
              .fontSize(13)
              .width(296)
              .height(60)
              .onClick(() => {
                this.resultForFile = '';
                this.info = '';
                this.selectFileBtnEnabled = false;

                const documentSelectOptions = new picker.DocumentSelectOptions();
                documentSelectOptions.maxSelectNumber = 1;
                documentSelectOptions.fileSuffixFilters = ['.wav'];
                const documentViewPicker = new picker.DocumentViewPicker();

                documentViewPicker.select(documentSelectOptions).then((result: Array<string>) => {
                  console.log(`select file result: ${result}`);

                  if (!result[0]) {
                    this.resultForFile = 'Please select a file to decode';
                    this.selectFileBtnEnabled = true;
                    return;
                  }

                  if (this.workerInstance) {
                    systemTime.getRealTime((err, data) => {
                      if (err) {
                        console.log('Failed to get start time');
                      } else {
                        this.startTime = data;
                      }
                    });

                    this.workerInstance.postMessage({
                      msgType: 'streaming-asr-decode-file', filename: result[0],
                    });
                    this.info = `Decoding ${result[0]} ... ...`;
                  } else {
                    console.log(`this worker instance is undefined ${this.workerInstance}`);
                  }

                }).catch((err: BusinessError) => {
                  console.error(`Failed to select file, code is ${err.code}, message is ${err.message}`);
                  this.selectFileBtnEnabled = true;
                })
              })

            Text(`Supported languages: ${this.lang}`);
            if (this.info != '') {
              TextArea({ text: this.info }).focusable(false);
            }
            TextArea({ text: this.resultForFile })
              .width('100%')
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .height('100%');
          }
        }.tabBar(this.TabBuilder('From file', 0, $r('app.media.icon_doc'), $r('app.media.icon_doc')))

        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
            Button(this.micBtnCaption)
              .enabled(this.micBtnEnabled)
              .fontSize(13)
              .width(296)
              .height(60)
              .onClick(() => {
                this.micInfo = '';
                if (this.mic) {
                  if (this.micStarted) {
                    this.micStarted = false;
                    this.micBtnCaption = 'Start';
                    this.mic.stop();
                    this.micSaveBtnEnabled = true;

                    if (this.workerInstance) {
                      this.workerInstance.postMessage({
                        msgType: 'streaming-asr-decode-mic-stop'
                      });
                    }
                  } else {
                    this.micStarted = true;
                    this.micSaveBtnEnabled = false;
                    this.micBtnCaption = 'Stop';
                    this.resultForMic = '';
                    this.resultListForMic = [];

                    if (this.workerInstance) {
                      this.workerInstance.postMessage({
                        msgType: 'streaming-asr-decode-mic-start'
                      });
                    }

                    this.sampleList = [];
                    this.mic.start();
                  }
                }
              });
            Button(this.micSaveBtnCaption)
              .enabled(this.micSaveBtnEnabled)
              .fontSize(13)
              .width(296)
              .height(60)
              .onClick(() => {
                if (this.sampleList.length == 0) {
                  this.micSaveBtnEnabled = false;
                  return;
                }

                const samples = this.flatten(this.sampleList);

                if (samples.length == 0) {
                  this.micSaveBtnEnabled = false;
                  return;
                }


                let uri: string = '';


                const audioOptions = new picker.AudioSaveOptions(); // audioOptions.newFileNames = ['o.wav'];

                const audioViewPicker = new picker.AudioViewPicker();

                audioViewPicker.save(audioOptions).then((audioSelectResult: Array<string>) => {
                  uri = audioSelectResult[0];
                  savePcmToWav(uri, toInt16Samples(samples), this.sampleRate);
                  console.log(`Saved to ${uri}`);
                  this.micInfo += `\nSaved to ${uri}`;
                });

              })


            Text(`Supported languages: ${this.lang}`)

            if (this.micInfo != '') {
              TextArea({ text: this.micInfo })
                .focusable(false);
            }

            TextArea({ text: this.resultForMic })
              .width('100%')
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .width('100%')
              .height('100%');
          }
        }.tabBar(this.TabBuilder('From mic', 1, $r('app.media.icon_mic'), $r('app.media.icon_mic')))


        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
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
    }.width('100%')
  }

  private micCallback = (buffer: ArrayBuffer) => {
    const view: Int16Array = new Int16Array(buffer);

    const samplesFloat: Float32Array = new Float32Array(view.length);
    for (let i = 0; i < view.length; ++i) {
      samplesFloat[i] = view[i] / 32768.0;
    }

    this.sampleList.push(samplesFloat);

    if (this.workerInstance) {
      this.workerInstance.postMessage({
        msgType: 'streaming-asr-decode-mic-samples',
        samples: samplesFloat,
        sampleRate: this.sampleRate,
      })
    }
  }
}