// Copyright (c)  2026  Xiaomi Corporation
// Web implementation of OfflineRecognizer using dart:js_interop.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';

import '../offline_recognizer_config.dart';
import 'init.dart';
import 'offline_stream.dart';

export '../offline_recognizer_config.dart';

/// Offline speech recognizer.
///
/// Create one from an [OfflineRecognizerConfig], then create an
/// [OfflineStream], feed waveform samples, call [decode], and fetch the final
/// hypothesis with [getResult].
class OfflineRecognizer {
  OfflineRecognizer._({required this.ptr, required this.config});

  factory OfflineRecognizer(OfflineRecognizerConfig config) {
    final m = getModule();

    // Use OfflineRecognizer class from sherpa-onnx-asr.js.
    final recognizerCtor =
        globalContext.getProperty('OfflineRecognizer'.toJS) as JSFunction?;
    if (recognizerCtor == null) {
      throw StateError(
          'OfflineRecognizer not found. Is sherpa-onnx-asr.js loaded?');
    }

    // Build config JSON matching JS wrapper expectations.
    final jsConfig = _buildConfig(config);

    final handle = recognizerCtor.callAsConstructor(jsConfig, m);
    if (handle == null) {
      throw Exception('Failed to create OfflineRecognizer');
    }

    return OfflineRecognizer._(ptr: handle, config: config);
  }

  final dynamic ptr;
  final OfflineRecognizerConfig config;
  bool _freed = false;

  void free() {
    if (_freed) return;
    final handle = ptr as JSObject;
    final freeFn = handle.getProperty('free'.toJS) as JSFunction?;
    freeFn?.callAsFunction(handle);
    _freed = true;
  }

  void setConfig(OfflineRecognizerConfig config) {
    final handle = ptr as JSObject;
    final setConfigFn = handle.getProperty('setConfig'.toJS) as JSFunction?;
    final jsConfig = _buildConfig(config);
    setConfigFn?.callAsFunction(handle, jsConfig);
  }

  OfflineStream createStream() {
    final handle = ptr as JSObject;
    final createStreamFn =
        handle.getProperty('createStream'.toJS) as JSFunction?;
    final streamHandle = createStreamFn?.callAsFunction(handle);
    if (streamHandle == null) {
      throw Exception('Failed to create OfflineStream');
    }
    return OfflineStream(ptr: streamHandle);
  }

  void decode(OfflineStream stream) {
    final handle = ptr as JSObject;
    final decodeFn = handle.getProperty('decode'.toJS) as JSFunction?;
    decodeFn?.callAsFunction(handle, stream.ptr as JSObject);
  }

  OfflineRecognizerResult getResult(OfflineStream stream) {
    final handle = ptr as JSObject;
    final getResultFn = handle.getProperty('getResult'.toJS) as JSFunction?;
    final jsResult = getResultFn?.callAsFunction(handle, stream.ptr as JSObject);
    if (jsResult == null) {
      return OfflineRecognizerResult(
          text: '', tokens: [], timestamps: [], lang: '', emotion: '', event: '');
    }

    final result = jsResult as JSObject;
    return OfflineRecognizerResult(
      text: (result.getProperty('text'.toJS) as JSString?)?.toDart ?? '',
      tokens: _jsArrayToStringList(result.getProperty('tokens'.toJS)),
      timestamps: _jsArrayToDoubleList(result.getProperty('timestamps'.toJS)),
      lang: (result.getProperty('lang'.toJS) as JSString?)?.toDart ?? '',
      emotion: (result.getProperty('emotion'.toJS) as JSString?)?.toDart ?? '',
      event: (result.getProperty('event'.toJS) as JSString?)?.toDart ?? '',
    );
  }
}

/// Build JS config object matching sherpa-onnx-asr.js expectations.
JSObject _buildConfig(OfflineRecognizerConfig config) {
  final jsConfig = JSObject();

  // Feature config.
  final feat = JSObject();
  feat['sampleRate'] = config.feat.sampleRate.toJS;
  feat['featureDim'] = config.feat.featureDim.toJS;
  jsConfig['featConfig'] = feat;

  // Model config.
  jsConfig['modelConfig'] = _buildModelConfig(config.model);

  // LM config.
  final lm = JSObject();
  lm['model'] = config.lm.model.toJS;
  lm['scale'] = config.lm.scale.toJS;
  jsConfig['lmConfig'] = lm;

  // Homophone replacer config.
  final hr = JSObject();
  hr['lexicon'] = config.hr.lexicon.toJS;
  hr['ruleFsts'] = config.hr.ruleFsts.toJS;
  jsConfig['hr'] = hr;

  // Scalar fields.
  jsConfig['decodingMethod'] = config.decodingMethod.toJS;
  jsConfig['maxActivePaths'] = config.maxActivePaths.toJS;
  jsConfig['hotwordsFile'] = config.hotwordsFile.toJS;
  jsConfig['hotwordsScore'] = config.hotwordsScore.toJS;
  jsConfig['ruleFsts'] = config.ruleFsts.toJS;
  jsConfig['ruleFars'] = config.ruleFars.toJS;
  jsConfig['blankPenalty'] = config.blankPenalty.toJS;

  return jsConfig;
}

JSObject _buildModelConfig(OfflineModelConfig model) {
  final jsModel = JSObject();

  // Transducer.
  final transducer = JSObject();
  transducer['encoder'] = model.transducer.encoder.toJS;
  transducer['decoder'] = model.transducer.decoder.toJS;
  transducer['joiner'] = model.transducer.joiner.toJS;
  jsModel['transducer'] = transducer;

  // Paraformer.
  final paraformer = JSObject();
  paraformer['model'] = model.paraformer.model.toJS;
  jsModel['paraformer'] = paraformer;

  // NeMo CTC.
  final nemoCtc = JSObject();
  nemoCtc['model'] = model.nemoCtc.model.toJS;
  jsModel['nemoCtc'] = nemoCtc;

  // Dolphin.
  final dolphin = JSObject();
  dolphin['model'] = model.dolphin.model.toJS;
  jsModel['dolphin'] = dolphin;

  // Zipformer CTC.
  final zipformerCtc = JSObject();
  zipformerCtc['model'] = model.zipformerCtc.model.toJS;
  jsModel['zipformerCtc'] = zipformerCtc;

  // Wenet CTC.
  final wenetCtc = JSObject();
  wenetCtc['model'] = model.wenetCtc.model.toJS;
  jsModel['wenetCtc'] = wenetCtc;

  // Omnilingual.
  final omnilingual = JSObject();
  omnilingual['model'] = model.omnilingual.model.toJS;
  jsModel['omnilingual'] = omnilingual;

  // MedASR.
  final medasr = JSObject();
  medasr['model'] = model.medasr.model.toJS;
  jsModel['medasr'] = medasr;

  // FireRed ASR CTC.
  final fireRedAsrCtc = JSObject();
  fireRedAsrCtc['model'] = model.fireRedAsrCtc.model.toJS;
  jsModel['fireRedAsrCtc'] = fireRedAsrCtc;

  // Whisper.
  final whisper = JSObject();
  whisper['encoder'] = model.whisper.encoder.toJS;
  whisper['decoder'] = model.whisper.decoder.toJS;
  whisper['language'] = model.whisper.language.toJS;
  whisper['task'] = model.whisper.task.toJS;
  whisper['tailPaddings'] = model.whisper.tailPaddings.toJS;
  whisper['enableTokenTimestamps'] = (model.whisper.enableTokenTimestamps ? 1 : 0).toJS;
  whisper['enableSegmentTimestamps'] = (model.whisper.enableSegmentTimestamps ? 1 : 0).toJS;
  jsModel['whisper'] = whisper;

  // TDNN.
  final tdnn = JSObject();
  tdnn['model'] = model.tdnn.model.toJS;
  jsModel['tdnn'] = tdnn;

  // SenseVoice.
  final senseVoice = JSObject();
  senseVoice['model'] = model.senseVoice.model.toJS;
  senseVoice['language'] = model.senseVoice.language.toJS;
  senseVoice['useInverseTextNormalization'] = (model.senseVoice.useInverseTextNormalization ? 1 : 0).toJS;
  jsModel['senseVoice'] = senseVoice;

  // Moonshine.
  final moonshine = JSObject();
  moonshine['preprocessor'] = model.moonshine.preprocessor.toJS;
  moonshine['encoder'] = model.moonshine.encoder.toJS;
  moonshine['uncachedDecoder'] = model.moonshine.uncachedDecoder.toJS;
  moonshine['cachedDecoder'] = model.moonshine.cachedDecoder.toJS;
  moonshine['mergedDecoder'] = model.moonshine.mergedDecoder.toJS;
  jsModel['moonshine'] = moonshine;

  // FireRed ASR.
  final fireRedAsr = JSObject();
  fireRedAsr['encoder'] = model.fireRedAsr.encoder.toJS;
  fireRedAsr['decoder'] = model.fireRedAsr.decoder.toJS;
  jsModel['fireRedAsr'] = fireRedAsr;

  // Canary.
  final canary = JSObject();
  canary['encoder'] = model.canary.encoder.toJS;
  canary['decoder'] = model.canary.decoder.toJS;
  canary['srcLang'] = model.canary.srcLang.toJS;
  canary['tgtLang'] = model.canary.tgtLang.toJS;
  canary['usePnc'] = (model.canary.usePnc ? 1 : 0).toJS;
  jsModel['canary'] = canary;

  // Qwen3 ASR.
  final qwen3Asr = JSObject();
  qwen3Asr['convFrontend'] = model.qwen3Asr.convFrontend.toJS;
  qwen3Asr['encoder'] = model.qwen3Asr.encoder.toJS;
  qwen3Asr['decoder'] = model.qwen3Asr.decoder.toJS;
  qwen3Asr['tokenizer'] = model.qwen3Asr.tokenizer.toJS;
  qwen3Asr['maxTotalLen'] = model.qwen3Asr.maxTotalLen.toJS;
  qwen3Asr['maxNewTokens'] = model.qwen3Asr.maxNewTokens.toJS;
  qwen3Asr['temperature'] = model.qwen3Asr.temperature.toJS;
  qwen3Asr['topP'] = model.qwen3Asr.topP.toJS;
  qwen3Asr['seed'] = model.qwen3Asr.seed.toJS;
  qwen3Asr['hotwords'] = model.qwen3Asr.hotwords.toJS;
  jsModel['qwen3Asr'] = qwen3Asr;

  // Cohere Transcribe.
  final cohereTranscribe = JSObject();
  cohereTranscribe['encoder'] = model.cohereTranscribe.encoder.toJS;
  cohereTranscribe['decoder'] = model.cohereTranscribe.decoder.toJS;
  cohereTranscribe['language'] = model.cohereTranscribe.language.toJS;
  cohereTranscribe['usePunct'] = (model.cohereTranscribe.usePunct ? 1 : 0).toJS;
  cohereTranscribe['useItn'] = (model.cohereTranscribe.useItn ? 1 : 0).toJS;
  jsModel['cohereTranscribe'] = cohereTranscribe;

  // FunASR Nano.
  final funasrNano = JSObject();
  funasrNano['encoderAdaptor'] = model.funasrNano.encoderAdaptor.toJS;
  funasrNano['llm'] = model.funasrNano.llm.toJS;
  funasrNano['embedding'] = model.funasrNano.embedding.toJS;
  funasrNano['tokenizer'] = model.funasrNano.tokenizer.toJS;
  funasrNano['systemPrompt'] = model.funasrNano.systemPrompt.toJS;
  funasrNano['userPrompt'] = model.funasrNano.userPrompt.toJS;
  funasrNano['maxNewTokens'] = model.funasrNano.maxNewTokens.toJS;
  funasrNano['temperature'] = model.funasrNano.temperature.toJS;
  funasrNano['topP'] = model.funasrNano.topP.toJS;
  funasrNano['seed'] = model.funasrNano.seed.toJS;
  funasrNano['language'] = model.funasrNano.language.toJS;
  funasrNano['itn'] = model.funasrNano.itn.toJS;
  funasrNano['hotwords'] = model.funasrNano.hotwords.toJS;
  jsModel['funasrNano'] = funasrNano;

  // Scalar fields.
  jsModel['tokens'] = model.tokens.toJS;
  jsModel['numThreads'] = model.numThreads.toJS;
  jsModel['debug'] = (model.debug ? 1 : 0).toJS;
  jsModel['provider'] = model.provider.toJS;
  jsModel['modelType'] = model.modelType.toJS;
  jsModel['modelingUnit'] = model.modelingUnit.toJS;
  jsModel['bpeVocab'] = model.bpeVocab.toJS;
  jsModel['telespeechCtc'] = model.telespeechCtc.toJS;

  return jsModel;
}

List<String> _jsArrayToStringList(JSAny? value) {
  if (value == null) return [];
  final arr = value as JSArray;
  final length = arr.getProperty('length'.toJS)!.dartify()! as int;
  final list = <String>[];
  for (int i = 0; i < length; i++) {
    list.add((arr.getProperty(i.toJS) as JSString).toDart);
  }
  return list;
}

List<double> _jsArrayToDoubleList(JSAny? value) {
  if (value == null) return [];
  final arr = value as JSArray;
  final length = arr.getProperty('length'.toJS)!.dartify()! as int;
  final list = <double>[];
  for (int i = 0; i < length; i++) {
    list.add((arr.getProperty(i.toJS) as JSNumber).toDartDouble);
  }
  return list;
}
