import { LengthUnit, promptAction } from '@kit.ArkUI';
import worker, { MessageEvents } from '@ohos.worker';
import { BusinessError, pasteboard } from '@kit.BasicServicesKit';
import { picker } from '@kit.CoreFileKit';


@Entry
@Component
struct Index {
  @State title: string = 'Next-gen Kaldi: Speaker Diarization';
  @State titleFontSize: number = 15;
  @State currentIndex: number = 0;
  @State resultForFile: string = '';
  @State resultForMic: string = '';
  @State progressForFile: number = 0;
  @State selectFileBtnEnabled: boolean = false;
  @State copyBtnForFileEnabled: boolean = false;
  private controller: TabsController = new TabsController();
  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/SpeakerDiarizationWorker.ets'
  private numSpeakers: string = '-1';

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

  aboutToAppear(): void {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'Streaming ASR worker'
    });

    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;

      if (msgType != 'speaker-diarization-file-progress') {
        console.log(`received msg from worker: ${msgType}`);
      }

      if (msgType == 'init-speaker-diarization-done') {
        console.log('Speaker diarization initialized successfully');

        this.resultForFile = 'Initialization finished.\nPlease select a .wav file.';
        this.resultForMic = 'Initialization finished.\nPlease click the button Start recording.';

        this.selectFileBtnEnabled = true;
      }

      if (msgType == 'speaker-diarization-file-progress') {
        this.progressForFile = e.data['progress'] as number;
      }

      if (msgType == 'speaker-diarization-file-done') {
        const result = e.data['result'] as string;
        this.resultForFile = result;

        this.selectFileBtnEnabled = true;
        this.copyBtnForFileEnabled = true;
      }
    };

    const context = getContext();
    this.workerInstance.postMessage({ msgType: 'init-speaker-diarization', context });
    console.log('initializing');
    this.resultForFile = 'Initializing models. Please wait';
    this.resultForMic = this.resultForFile;
  }

  build() {
    Column() {
      Tabs({ barPosition: BarPosition.End, controller: this.controller }) {
        TabContent() {
          Column({ space: 10 }) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
            Row({ space: 10 }) {
              Text(`Number of speakers`).width('60%')

              TextInput({ text: this.numSpeakers }).onChange((text) => {
                this.numSpeakers = text.trim();
              }).width('20%')
            }.justifyContent(FlexAlign.Center)

            Row({ space: 10 }) {
              Button('Select .wav file (16kHz) ').enabled(this.selectFileBtnEnabled).onClick(() => {
                this.resultForFile = '';
                this.progressForFile = 0;
                this.copyBtnForFileEnabled = false;

                let numSpeakers = parseInt(this.numSpeakers);
                if (numSpeakers.toString() != this.numSpeakers) {
                  this.resultForFile =
                    'Please input a valid value for the number of speakers in the .wav file you are going to select';
                  return;
                }

                if (numSpeakers < 1) {
                  this.resultForFile =
                    'Please input a positive value for the number of speakers in the .wav file you are going to select';
                  return;
                }

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
                    this.workerInstance.postMessage({
                      msgType: 'speaker-diarization-file', filename: result[0], numSpeakers,
                    });
                    this.resultForFile = `Decoding ${result[0]} ... ...`;
                  } else {
                    console.log(`this worker instance is undefined ${this.workerInstance}`);
                  }
                }).catch((err: BusinessError) => {
                  console.error(`Failed to select file, code is ${err.code}, message is ${err.message}`);
                  this.selectFileBtnEnabled = true;
                })
              })
              Button('Copy results')
                .enabled(this.copyBtnForFileEnabled)
                .onClick(() => { // See https://developer.huawei.com/consumer/cn/doc/harmonyos-faqs/faqs-arkui-308-V5
                  const pasteboardData = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, this.resultForFile);
                  const systemPasteboard = pasteboard.getSystemPasteboard();
                  systemPasteboard.setData(pasteboardData);
                  systemPasteboard.getData().then((data) => {
                    if (data) {
                      promptAction.showToast({ message: 'Result copied.' });
                    } else {
                      promptAction.showToast({ message: 'Failed to copy' });
                    }
                  })
                })
            }

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
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .width('100%')
              .height('100%')
          }
        }.tabBar(this.TabBuilder('From file', 0, $r('app.media.icon_doc'), $r('app.media.icon_doc')))

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
        }.tabBar(this.TabBuilder('Help', 1, $r('app.media.info'), $r('app.media.info')))
      }.scrollable(false)
    }
  }
}