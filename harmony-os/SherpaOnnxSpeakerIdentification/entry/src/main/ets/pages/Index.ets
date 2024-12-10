import worker, { MessageEvents } from '@ohos.worker';

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

  aboutToAppear() {
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
            if (this.allSpeakerNames.length == 0) {
              Text('Please add speakers first')
                .fontSize(30)
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
                          console.log(`Deleting speaker ${name}}`);
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
            Button('Add')
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
}