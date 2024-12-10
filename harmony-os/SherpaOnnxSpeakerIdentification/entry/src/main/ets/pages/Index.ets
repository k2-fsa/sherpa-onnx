import worker, { MessageEvents } from '@ohos.worker';
import { audio } from '@kit.AudioKit';
import { allAllowed, requestPermissions } from './Permission';
import { Permissions } from '@kit.AbilityKit';
import { picker } from '@kit.CoreFileKit';
import fs from '@ohos.file.fs';



function flatten(samples: Float32Array[]): Float32Array {
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
  @State title: string = 'Next-gen Kaldi: Speaker Identification';
  @State titleFontSize: number = 18;
  private controller: TabsController = new TabsController();

  @State currentIndex: number = 0;

  @State message: string = 'Hello World';

  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/SpeakerIdentificationWorker.ets'

  @State allSpeakerNames: string[] = [];
  private inputSpeakerName: string = '';

  @State btnSaveAudioEnabled: boolean = false;
  @State btnAddEnabled: boolean = false;

  private sampleRate: number = 16000;
  private sampleList: Float32Array[] = []
  private mic?: audio.AudioCapturer;

  @State infoHome: string = '';
  @State infoAdd: string = '';

  @State micBtnCaption: string = 'Start recording';
  @State micStarted: boolean = false;

  async initMic() {
    const permissions: Permissions[] = ["ohos.permission.MICROPHONE"];
    let allowed: boolean = await allAllowed(permissions);
    if (!allowed) {
      console.log("request to access the microphone");
      const status: boolean = await requestPermissions(permissions);

      if (!status) {
        console.error('access to microphone is denied')
        this.infoHome = "Failed to get microphone permission. Please retry";
        this.infoAdd = this.infoHome;
        return;
      }

      allowed = await allAllowed(permissions);
      if (!allowed) {
        console.error('failed to get microphone permission');
        this.infoHome = "Failed to get microphone permission. Please retry";
        this.infoAdd = this.infoHome;
        return;
      }
    } else {
      console.log("allowed to access microphone");
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
        this.infoHome = 'Failed to init microphone';
        this.infoAdd = this.infoHome;
      } else {
        console.info(`init mic successfully`);
        this.mic = data;
        this.mic.on('readData', this.micCallback);
      }
    });
  }

  async aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'Speaker identification worker'
    });

    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'manager-all-speaker-names') {
        this.allSpeakerNames = e.data['allSpeakers'] as string[];
      }
    };

    this.workerInstance.postMessage({ msgType: 'init-extractor', context: getContext()});

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
            Button('Home')
          }
        }.tabBar(this.TabBuilder('Home', 0, $r('app.media.icon_home'), $r('app.media.icon_home')))

        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);

            if (this.allSpeakerNames.length == 0) {
              Text('Please add speakers first')
            } else {
              List({ space: 10, initialIndex: 0 }) {
                ForEach(this.allSpeakerNames, (item: string, index: number) => {
                  ListItem() {
                    Flex({ direction: FlexDirection.Row, alignItems: ItemAlign.Center }) {
                      Text(item)
                        .width('100%')
                        .height(80)
                        .fontSize(20)
                        .textAlign(TextAlign.Center)
                        .borderRadius(10)
                        .flexShrink(1)

                      Button('Delete')
                      .width('30%')
                        .height(40)
                      .onClick(() => {
                        if (index != undefined) {
                          const name = this.allSpeakerNames[index];
                          console.log(`Deleting speaker ${name}`);
                          if (this.workerInstance) {
                            this.workerInstance.postMessage({
                              msgType: 'manager-delete-speaker',
                              name: name
                            });
                          }
                        }
                      }).stateEffect(true)

                      Text('')
                        .width('15%')
                        .height(80)
                    }
                  }
                }, (item: string) => item)
              }
            }
          }
        }.tabBar(this.TabBuilder('View', 1, $r('app.media.icon_view'), $r('app.media.icon_view')))

        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);

            Row({space: 10}) {
              Text('Speaker name')
              TextInput({placeholder: 'Input speaker name'})
                .onChange((value: string)=>{
                  this.inputSpeakerName = value.trim();
                });
            }.width('100%')

            Row({space: 10}) {
              Button(this.micBtnCaption)
                .onClick(()=> {
                  if (this.mic) {
                    if (this.micStarted) {
                      this.micStarted = false;
                      this.micBtnCaption = 'Start recording';
                      this.mic.stop();
                      this.infoAdd = '';
                      if (this.sampleList.length > 0) {
                        this.btnAddEnabled = true;
                        this.btnSaveAudioEnabled = true;
                      }
                    } else {
                      this.micStarted = true;
                      this.micBtnCaption = 'Stop recording';
                      this.sampleList = [];
                      this.mic.start();
                      this.infoAdd = '';

                      this.btnAddEnabled = false;
                      this.btnSaveAudioEnabled = false;
                    }
                  }

                })

              Button('Add')
                .enabled(this.btnAddEnabled)
                .onClick(()=>{
                  if (this.inputSpeakerName.trim() == '') {
                    this.infoAdd += 'Please input a speaker name first';
                    return;
                  }

                  const samples = flatten(this.sampleList);
                  console.log(`number of samples: ${samples.length}, ${samples.length / this.sampleRate}`);
                })

              Button('Save audio')
                .enabled(this.btnSaveAudioEnabled)
                .onClick(()=>{
                  if (this.sampleList.length == 0) {
                    this.btnSaveAudioEnabled = false;
                    return;
                  }

                  const samples = flatten(this.sampleList);

                  if (samples.length == 0) {
                    this.btnSaveAudioEnabled = false;
                    return;
                  }

                  let uri: string = '';


                  const audioOptions = new picker.AudioSaveOptions(); // audioOptions.newFileNames = ['o.wav'];

                  const audioViewPicker = new picker.AudioViewPicker();

                  audioViewPicker.save(audioOptions).then((audioSelectResult: Array<string>) => {
                    uri = audioSelectResult[0];
                    savePcmToWav(uri, toInt16Samples(samples), this.sampleRate);
                    console.log(`Saved to ${uri}`);
                    this.infoAdd += `\nSaved to ${uri}`;
                  });
                })
            }
            TextArea({text: this.infoAdd})
              .height('100%')
              .width('100%')
              .focusable(false)
          }
        }.tabBar(this.TabBuilder('Add', 2, $r('app.media.icon_add'), $r('app.media.icon_add')))

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
          }
        }.tabBar(this.TabBuilder('Help', 3, $r('app.media.icon_info'), $r('app.media.icon_info')))

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
  }
}