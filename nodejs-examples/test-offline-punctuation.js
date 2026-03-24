// Copyright (c)  2026  Xiaomi Corporation
//
const sherpa_onnx = require('sherpa-onnx');

function createOfflinePunctuation() {
  const config = {
    model: {
      ctTransformer:
          './sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx',
      debug: 1,
      numThreads: 1,
      provider: 'cpu',
    },
  };

  return sherpa_onnx.createOfflinePunctuation(config);
}

const punct = createOfflinePunctuation();
const sentences = [
  '这是一个测试你好吗How are you我很好thank you are you ok谢谢你',
  '我们都是木头人不会说话不会动',
  'The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry',
];

console.log('---');
for (const sentence of sentences) {
  const punctText = punct.addPunct(sentence);
  console.log(`Input: ${sentence}`);
  console.log(`Output: ${punctText}`);
  console.log('---');
}

punct.free();
