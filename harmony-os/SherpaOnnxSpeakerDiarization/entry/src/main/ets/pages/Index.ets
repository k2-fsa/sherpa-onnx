import { LengthUnit } from '@kit.ArkUI';

@Entry
@Component
struct Index {
  @State title: string = 'Next-gen Kaldi: Speaker Diarization';
  @State titleFontSize: number = 15;
  @State currentIndex: number = 0;
  private controller: TabsController = new TabsController();

  @State resultForFile: string = '';
  @State resultForMic: string = '';

  @State micBtnCaption: string = 'Start recording';
  @State micSaveBtnCaption: string = 'Save';

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
          Column({space: 10}) {
            Text(this.title).fontSize(this.titleFontSize).fontWeight(FontWeight.Bold);
            Button('Select .wav file (16kHz) ')

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