import { LengthUnit } from '@kit.ArkUI';
import worker, { MessageEvents } from '@ohos.worker';
import { BusinessError } from '@kit.BasicServicesKit';
import { picker } from '@kit.CoreFileKit';
import systemTime from '@ohos.systemTime';


@Entry
@Component
struct Index {
  @State title: string = 'Next-gen Kaldi: Real-time speech recognition';
  @State titleFontSize: number = 15;
  @State currentIndex: number = 0;
  @State lang: string = 'English';
  @State resultForFile: string = ''
  @State resultForMic: string = ''
  @State micBtnCaption: string = 'Start';
  @State selectFileBtnEnabled: boolean = false;
  @State info: string = '';
  private controller: TabsController = new TabsController();
  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/StreamingAsrWorker.ets'
  private startTime: number = 0;
  private stopTime: number = 0;

  aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'Streaming ASR worker'
    });

    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received omsg from worker: ${msgType}`);

      if (msgType == 'init-streaming-asr-done') {
        this.selectFileBtnEnabled = true;
        this.info = `Initializing done.\n\nPlease select a wave file of 16kHz in language ${this.lang}`;
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
    };

    const context = getContext();
    this.workerInstance.postMessage({ msgType: 'init-streaming-asr', context });
    this.info = 'Intializing ASR model.\nPlease wait';
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
            Text(`Supported languages: ${this.lang}`)
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
}