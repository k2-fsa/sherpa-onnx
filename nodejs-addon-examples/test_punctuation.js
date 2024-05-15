// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx-node');

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
function createPunctuation() {
  const config = {
    model: {
      ctTransformer:
          './sherpa-onnx-punct-ct-transformer-zh-en-vocab272727-2024-04-12/model.onnx',
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
  };
  return new sherpa_onnx.Punctuation(config);
}

const punct = createPunctuation();
const sentences = [
  '这是一个测试你好吗How are you我很好thank you are you ok谢谢你',
  '我们都是木头人不会说话不会动',
  'The African blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry',
];
console.log('---');
for (let sentence of sentences) {
  const punct_text = punct.addPunct(sentence);
  console.log(`Input: ${sentence}`);
  console.log(`Output: ${punct_text}`);
  console.log('---');
}
