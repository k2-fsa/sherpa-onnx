// Copyright (c)  2026  Xiaomi Corporation
const fs = require('fs');
const sherpa_onnx = require('sherpa-onnx-node');

// This example shows how to use per-stream hotwords (contextual biasing)
// with a NeMo transducer model. Hotwords require
// decodingMethod 'modified_beam_search'.
//
// The test wave contains the name "Phoebe", which the model transcribes
// with the spelling "Phebe" by default. Passing the hotword "Phoebe" to
// createStream() biases this stream towards the expected spelling.
//
// Please download test files from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models

const modelDir = './sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8';

// To pass hotwords as normal words, the model config needs a bpe vocab.
// This model's release does not ship bpe.vocab, but an equivalent one for
// hotword encoding can be derived from tokens.txt (equal scores make the
// encoder behave as longest-match).
const bpeVocab = `${modelDir}/bpe.vocab`;
if (!fs.existsSync(bpeVocab)) {
  const tokens = fs.readFileSync(`${modelDir}/tokens.txt`, 'utf8');
  const vocab = tokens.split('\n')
                    .filter(line => line.trim() !== '')
                    .map(line => `${line.split(' ')[0]}\t-1.0`)
                    .join('\n');
  fs.writeFileSync(bpeVocab, vocab + '\n');
}

const config = {
  'featConfig': {
    'sampleRate': 16000,
    'featureDim': 80,
  },
  'modelConfig': {
    'transducer': {
      'encoder':
          './sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/encoder.int8.onnx',
      'decoder':
          './sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/decoder.int8.onnx',
      'joiner': './sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/joiner.int8.onnx',
    },
    'tokens': './sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/tokens.txt',
    'numThreads': 2,
    'provider': 'cpu',
    'debug': 1,
    'modelType': 'nemo_transducer',
    'modelingUnit': 'bpe',
    'bpeVocab': bpeVocab,
  },
  'decodingMethod': 'modified_beam_search',
  'hotwordsScore': 2.0,
};

const waveFilename =
    './sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8/test_wavs/0.wav';

const recognizer = new sherpa_onnx.OfflineRecognizer(config);
console.log('Started');
const wave = sherpa_onnx.readWave(waveFilename);

function decode(hotwords) {
  const stream = hotwords === undefined ? recognizer.createStream() :
                                          recognizer.createStream(hotwords);
  stream.acceptWaveform({sampleRate: wave.sampleRate, samples: wave.samples});
  recognizer.decode(stream);
  return recognizer.getResult(stream).text;
}

console.log('Without hotwords:', decode());

// Multiple phrases are separated by '/'; a per-phrase boosting score can be
// appended, e.g. 'PHOEBE :3.0/DON QUIXOTE'.
console.log('With hotwords   :', decode('Phoebe'));
