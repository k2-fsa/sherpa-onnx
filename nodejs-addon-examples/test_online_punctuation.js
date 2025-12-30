// Copyright (c)  2023-2024  Xiaomi Corporation (authors: Fangjun Kuang)

const sherpa_onnx = require('sherpa-onnx-node');

// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models
function createPunctuation() {
  const config = {
    model: {
      cnnBilstm:
          './sherpa-onnx-online-punct-en-2024-08-06/model.onnx',
      bpeVocab:
          './sherpa-onnx-online-punct-en-2024-08-06/bpe.vocab',
      debug: true,
      numThreads: 1,
      provider: 'cpu',
    },
  };
  return new sherpa_onnx.OnlinePunctuation(config);
}

const punct = createPunctuation();
const sentences = [
  'How are you i am fine thank you',
  'The african blogosphere is rapidly expanding bringing more voices online in the form of commentaries opinions analyses rants and poetry',
];
console.log('---');
for (let sentence of sentences) {
  const punct_text = punct.addPunct(sentence);
  console.log(`Input: ${sentence}`);
  console.log(`Output: ${punct_text}`);
  console.log('---');
}
