import worker, { MessageEvents } from '@ohos.worker';

@Entry
@Component


struct Index {
  @State message: string = 'Hello World';

  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/SpeakerIdentificationWorker.ets'

  aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'Speaker identification worker'
    });
  }

  build() {
    Row() {
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(()=>{
            if (this.workerInstance) {
              const context = getContext();
              this.workerInstance.postMessage({ msgType: 'init-extractor', context });
            }
            console.log('clicked');
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}