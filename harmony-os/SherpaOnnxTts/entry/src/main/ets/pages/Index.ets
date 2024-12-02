import { listRawfileDir } from 'sherpa_onnx';
import worker, { MessageEvents } from '@ohos.worker';


@Entry
@Component
struct Index {
  @State message: string = 'Hello World';

  private workerInstance?: worker.ThreadWorker
  private readonly scriptURL: string = 'entry/ets/workers/NonStreamingTtsWorker.ets'

  aboutToAppear() {
    this.workerInstance = new worker.ThreadWorker(this.scriptURL, {
      name: 'NonStreaming TTS worker'
    });
    this.workerInstance.onmessage = (e: MessageEvents) => {
      const msgType = e.data['msgType'] as string;
      console.log(`received msg from worker: ${msgType}`);

      if (msgType == 'init-tts-done') {
        console.log('init tts done');
      }
    }

    this.workerInstance.postMessage({ msgType: 'init-tts', context: getContext() });
  }

  build() {
    Row() {
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(()=>{
            console.log("clicked");
            const mgr = getContext().resourceManager;
            const files: string[] = listRawfileDir(mgr, "");
            for (const f of files) {
              console.log(f);
            }
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}