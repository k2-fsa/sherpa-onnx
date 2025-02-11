// Please keep in sync with
// https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/kotlin-api/OfflineRecognizer.kt#L184

import { OfflineModelConfig } from 'sherpa_onnx';

export function getOfflineModelConfig(type: number): OfflineModelConfig {
  const c: OfflineModelConfig = new OfflineModelConfig();
  switch (type) {
    case 0: {
      const modelDir = 'sherpa-onnx-paraformer-zh-2023-09-14'
      c.paraformer.model = `${modelDir}/model.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "paraformer";

      break;
    }

    case 1: {
      const modelDir = 'icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04'
      c.transducer.encoder = `$modelDir}/encoder-epoch-30-avg-4.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-30-avg-4.onnx`;
      c.transducer.encoder = `${modelDir}/joiner-epoch-30-avg-4.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 2: {
      const modelDir = 'sherpa-onnx-whisper-tiny.en';
      c.whisper.encoder = `${modelDir}/tiny.en-encoder.int8.onnx`;
      c.whisper.decoder = `${modelDir}/tiny.en-decoder.int8.onnx`;
      c.tokens = `${modelDir}/tiny.en-tokens.txt`;
      c.modelType = "whisper";

      break;
    }

    case 3: {
      const modelDir = 'sherpa-onnx-whisper-base.en';
      c.whisper.encoder = `${modelDir}/base.en-encoder.int8.onnx`;
      c.whisper.decoder = `${modelDir}/base.en-decoder.int8.onnx`;
      c.tokens = `${modelDir}/base.en-tokens.txt`;
      c.modelType = "whisper";

      break;
    }

    case 4: {
      const modelDir = "icefall-asr-zipformer-wenetspeech-20230615";
      c.transducer.encoder = `${modelDir}/encoder-epoch-12-avg-4.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-12-avg-4.onnx`;
      c.transducer.joiner = `${modelDir}/joiner-epoch-12-avg-4.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 5: {
      const modelDir = "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2";
      c.transducer.encoder = `${modelDir}/encoder-epoch-20-avg-1.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-20-avg-1.onnx`;
      c.transducer.joiner = `${modelDir}/joiner-epoch-20-avg-1.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 6: {
      const modelDir = "sherpa-onnx-nemo-ctc-en-citrinet-512";
      c.nemoCtc.model = `${modelDir}/model.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 7: {
      const modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k"
      c.nemoCtc.model = `${modelDir}/model.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 8: {
      const modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-en-24500"
      c.nemoCtc.model = `${modelDir}/model.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 9: {
      const modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288"
      c.nemoCtc.model = `${modelDir}/model.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 10: {
      const modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-es-1424"
      c.nemoCtc.model = `${modelDir}/model.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 11: {
      const modelDir = "sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04"
      c.telespeechCtc = `${modelDir}/model.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "telespeech_ctc";

      break;
    }

    case 12: {
      const modelDir = "sherpa-onnx-zipformer-thai-2024-06-20"
      c.transducer.encoder = `${modelDir}/encoder-epoch-12-avg-5.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-12-avg-5.onnx`;
      c.transducer.joiner = `${modelDir}/joiner-epoch-12-avg-5.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 13: {
      const modelDir = "sherpa-onnx-zipformer-korean-2024-06-24";
      c.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      c.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 14: {
      const modelDir = "sherpa-onnx-paraformer-zh-small-2024-03-09";
      c.paraformer.model = `${modelDir}/model.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "paraformer";

      break;
    }

    case 15: {
      const modelDir = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
      c.senseVoice.model = `${modelDir}/model.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 16: {
      const modelDir = "sherpa-onnx-zipformer-ja-reazonspeech-2024-08-01";
      c.transducer.encoder = `${modelDir}/encoder-epoch-99-avg-1.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-99-avg-1.onnx`;
      c.transducer.joiner = `${modelDir}/joiner-epoch-99-avg-1.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 17: {
      const modelDir = "sherpa-onnx-zipformer-ru-2024-09-18";
      c.transducer.encoder = `${modelDir}/encoder.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder.onnx`;
      c.transducer.joiner = `${modelDir}/joiner.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 18: {
      const modelDir = "sherpa-onnx-small-zipformer-ru-2024-09-18";
      c.transducer.encoder = `${modelDir}/encoder.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder.onnx`;
      c.transducer.joiner = `${modelDir}/joiner.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    case 19: {
      const modelDir = "sherpa-onnx-nemo-ctc-giga-am-russian-2024-10-24";
      c.nemoCtc.model = `${modelDir}/model.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 20: {
      const modelDir = "sherpa-onnx-nemo-transducer-giga-am-russian-2024-10-24";
      c.transducer.encoder = `${modelDir}/encoder.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder.onnx`;
      c.transducer.joiner = `${modelDir}/joiner.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "nemo_transducer";

      break;
    }

    case 21: {
      const modelDir = "sherpa-onnx-moonshine-tiny-en-int8";
      c.moonshine.preprocessor = `${modelDir}/preprocess.onnx`;
      c.moonshine.encoder = `${modelDir}/encode.int8.onnx`;
      c.moonshine.uncachedDecoder = `${modelDir}/uncached_decode.int8.onnx`;
      c.moonshine.cachedDecoder = `${modelDir}/cached_decode.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 22: {
      const modelDir = "sherpa-onnx-moonshine-base-en-int8";
      c.moonshine.preprocessor = `${modelDir}/preprocess.onnx`;
      c.moonshine.encoder = `${modelDir}/encode.int8.onnx`;
      c.moonshine.uncachedDecoder = `${modelDir}/uncached_decode.int8.onnx`;
      c.moonshine.cachedDecoder = `${modelDir}/cached_decode.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;

      break;
    }

    case 23: {
      const modelDir = "sherpa-onnx-zipformer-zh-en-2023-11-22";
      c.transducer.encoder = `${modelDir}/encoder-epoch-34-avg-19.int8.onnx`;
      c.transducer.decoder = `${modelDir}/decoder-epoch-34-avg-19.onnx`;
      c.transducer.joiner = `${modelDir}/joiner-epoch-34-avg-19.int8.onnx`;
      c.tokens = `${modelDir}/tokens.txt`;
      c.modelType = "transducer";

      break;
    }

    default: {
      console.log(`Please specify a supported type. Given type ${type}`);
    }
  }

  return c;
}
