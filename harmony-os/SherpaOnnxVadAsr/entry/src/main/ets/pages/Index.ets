import { LengthUnit } from '@kit.ArkUI';

@Entry
@Component
struct Index {
  @State currentIndex: number = 0;
  private controller: TabsController = new TabsController();
  @State resultFromFile: string = '';
  @State progressForFile: number = 0;
  @State selectFileBtnEnabled: boolean = true;
  @State message: string = 'To be implemented';

  @Builder
  TabBuilder(title: string, targetIndex: number, selectedImg: Resource, normalImg: Resource) {
    Column() {
      Image(this.currentIndex == targetIndex ? selectedImg : normalImg)
        .size({ width: 25, height: 25 })
      Text(title)
        .fontColor(this.currentIndex == targetIndex ? '#28bff1' : '#8a8a8a')
    }
    .width('100%')
    .height(50)
    .justifyContent(FlexAlign.Center)
    .onClick(() => {
      this.currentIndex = targetIndex;
      this.controller.changeIndex(this.currentIndex);
    })
  }

  build() {
    Column() {
      Tabs({ barPosition: BarPosition.End, controller: this.controller }) {
        TabContent() {
          Column({ space: 10 }) {
            Text('Next-gen Kaldi: VAD + ASR')
              .fontColor('#182431')
              .fontSize(25)
              .lineHeight(41)
              .fontWeight(500)

            Button('Select .wav file ')
              .enabled(this.selectFileBtnEnabled)
              .fontSize(13)
              .width(296)
              .height(60)
              .onClick(() => {

              })

            if (this.progressForFile > 0) {
              Row() {
                Progress({ value: 0, total: 100, type: ProgressType.Capsule })
                  .width('80%')
                  .height(20)
                  .value(this.progressForFile);

                Text(`${this.progressForFile.toFixed(2)}%`).width('15%')
              }.width('100%').justifyContent(FlexAlign.Center)
            }

            TextArea({ text: this.resultFromFile }).width('100%').lineSpacing({ value: 10, unit: LengthUnit.VP });

          }
          .alignItems(HorizontalAlign.Center)
          .justifyContent(FlexAlign.Start)
        }.tabBar(this.TabBuilder('From file', 0, $r('app.media.icon_doc'), $r('app.media.icon_doc_default')))

        TabContent() {
          Column() {
            Text(this.message)
              .fontSize(50)
              .fontWeight(FontWeight.Bold);
          }
        }
        .tabBar(this.TabBuilder('From mic', 1, $r('app.media.ic_public_input_voice'),
          $r('app.media.ic_public_input_voice_default')))

        TabContent() {
          Column() {
            Text("Everything is open-sourced");
            Divider();
            Text("It runs locally, without accessing the network");
            Divider();
            Text("See also https://github.com/k2-fsa/sherpa-onnx");
            Divider();
            Text("and https://k2-fsa.github.io/sherpa/social-groups.html");
          }
        }.tabBar(this.TabBuilder('Help', 2, $r('app.media.info_circle'),
          $r('app.media.info_circle_default')))

      }.scrollable(false)
    }
    .width('100%')
    .justifyContent(FlexAlign.Start)
  }
}