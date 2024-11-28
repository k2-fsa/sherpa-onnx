import hilog from '@ohos.hilog';
import testNapi from 'libsherpa_onnx.so';

@Component
export struct MainPage {
  @State message: string = 'Hello World';

  build() {
    Row() {
      Column() {
        Text(this.message)
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(() => {
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}