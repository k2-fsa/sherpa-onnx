import { OfflinePunctuation, OfflinePunctuationModelConfig } from 'sherpa_onnx';

@Entry
@Component
struct Index {
  test(){
    const modelConfig = new OfflinePunctuationModelConfig();
    const punctModelDir = 'sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12-int8';
    modelConfig.ctTransformer = `${punctModelDir}/model.int8.onnx`;
    let punctuation: OfflinePunctuation = new OfflinePunctuation(modelConfig)
    let text = punctuation.offlinePunctuationAddPunct('这是一个测试你好吗How are you我很好thank you are you ok谢谢你')
  }

  build() {
    Row() {
      Column() {
        Text('测试，按下这里')
          .fontSize(50)
          .fontWeight(FontWeight.Bold)
          .onClick(()=>{
            this.test()
          })
      }
      .width('100%')
    }
    .height('100%')
  }
}