import { LengthUnit } from '@kit.ArkUI';
import worker, { MessageEvents } from '@ohos.worker';
import { systemTime, BusinessError } from '@kit.BasicServicesKit';
import { picker } from '@kit.CoreFileKit';


@Entry
@Component
struct Index {
  @State title: string = 'Next-gen Kaldi: Speaker Diarization';
  @State titleFontSize: number = 15;
  @State currentIndex: number = 0;
  private controller: TabsController = new TabsController();

  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/SpeakerDiarizationWorker.ets'

  @State resultForFile: string = '';
  @State resultForMic: string = '';

  @State micBtnCaption: string = 'Start recording';
  @State micSaveBtnCaption: string = 'Save audio';

  @State micBtnEnabled: boolean = false;
  @State selectFileBtnEnabled: boolean = false;

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
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'init-speaker-diarization-done') {
        console.log('Speaker diarization initialized successfully');

        this.resultForFile = 'Initialization finished.\nPlease select a .wav file.';
        this.resultForMic = 'Initialization finished.\nPlease click the button Start recording.';

        this.selectFileBtnEnabled = true;
        this.micBtnEnabled = true;
      }

      if (msgType == 'speaker-diarization-file-done') {
        const result = e.data['result'] as string;
        this.resultForFile = result;

        this.selectFileBtnEnabled = true;
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
          Column({space: 10}) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
            Button('Select .wav file (16kHz) ')
              .enabled(this.selectFileBtnEnabled)
              .onClick(()=>{
                this.resultForFile = '';
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
                      msgType: 'speaker-diarization-file', filename: result[0],
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

            TextArea({ text: this.resultForFile })
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .width('100%')
              .height('100%')
          }
        }.tabBar(this.TabBuilder('From file', 0, $r('app.media.icon_doc'), $r('app.media.icon_doc')))

        TabContent() {
          Column({space: 10}) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
            Row() {
              Button(this.micBtnCaption)

              Button(this.micSaveBtnCaption)
            }
            TextArea({ text: this.resultForMic })
              .lineSpacing({ value: 10, unit: LengthUnit.VP })
              .width('100%')
              .height('100%')
          }
        }.tabBar(this.TabBuilder('From mic', 1, $r('app.media.icon_mic'), $r('app.media.icon_mic')))

        TabContent() {
          Column({space: 10}) {
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
    }
  }
}