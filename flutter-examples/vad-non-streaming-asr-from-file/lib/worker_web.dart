// Web Worker support for VAD+ASR.
import 'dart:async';
import 'dart:convert';
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show kDebugMode;
import 'package:flutter/services.dart';
import 'package:web/web.dart' as web;

import './model_web.dart' as m;
import './model_config.dart' as cfg;
import './vad_asr_manager_web.dart' show VadAsrSegment;

typedef OnReadyCallback = void Function();
typedef OnStartedCallback = void Function(int runId);
typedef OnProgressCallback = void Function(double progress);
typedef OnSegmentCallback = void Function(VadAsrSegment segment);
typedef OnResultCallback = void Function(
    List<VadAsrSegment> segments, double elapsed, double audioDuration);
typedef OnErrorCallback = void Function(String message);

/// Manages a Web Worker for VAD+ASR.
class VadAsrWorker {
  web.Worker? _worker;
  int _currentRunId = 0;
  final OnReadyCallback onReady;
  final OnStartedCallback onStarted;
  final OnProgressCallback onProgress;
  final OnSegmentCallback onSegment;
  final OnResultCallback onResult;
  final OnErrorCallback onError;

  VadAsrWorker({
    required this.onReady,
    required this.onStarted,
    required this.onProgress,
    required this.onSegment,
    required this.onResult,
    required this.onError,
  });

  /// Initialize the worker: load WASM and model files, send to worker.
  Future<void> init() async {
    final vadModelFiles = await m.loadModelFileBytes();
    final vadConfig = await m.prepareModelConfig();

    // Load ASR model files for the selected model.
    final asrModelDir = cfg.selectedAsrModel.assetFiles[0]
        .substring(0, cfg.selectedAsrModel.assetFiles[0].lastIndexOf('/'));
    final asrModelFiles = <String, Uint8List>{};
    final assetManifest = await AssetManifest.loadFromAssetBundle(rootBundle);
    final allAssets = assetManifest.listAssets();
    for (final asset in allAssets) {
      if (asset.contains(asrModelDir)) {
        final bytes = await _loadAssetBytes(asset);
        final relativePath = asset.replaceFirst('assets/', '');
        asrModelFiles[relativePath] = bytes;
      }
    }

    // Build ASR config for the web worker.
    final asrConfig = _buildAsrConfigForWeb();

    if (kDebugMode) {
      print('[worker_web] VAD model files: ${vadModelFiles.length}');
      print('[worker_web] ASR model files: ${asrModelFiles.length}');
    }

    // Create Web Worker.
    _worker = web.Worker('vad-asr-worker.js'.toJS);

    _worker!.onmessage = (web.MessageEvent event) {
      _handleMessage(event);
    }.toJS;

    _worker!.onerror = (web.ErrorEvent event) {
      onError('Worker error: ${event.message}');
    }.toJS;

    // Load JS sources and WASM binary.
    final jsGlueSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.js');
    final vadJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-vad.js');
    final asrJsSource = await _loadAssetAsString(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-asr.js');
    final wasmData = await _loadAssetBytes(
        'packages/sherpa_onnx_web/assets/sherpa-onnx-wasm-web.wasm');

    // Build model files maps.
    final jsVadModelFiles = JSObject();
    for (final entry in vadModelFiles.entries) {
      jsVadModelFiles[entry.key] = Uint8List.fromList(entry.value).buffer.toJS;
    }

    final jsAsrModelFiles = JSObject();
    for (final entry in asrModelFiles.entries) {
      jsAsrModelFiles[entry.key] = entry.value.buffer.toJS;
    }

    // Send init message.
    final initMsg = JSObject();
    initMsg['type'] = 'init'.toJS;
    initMsg['jsGlueSource'] = jsGlueSource.toJS;
    initMsg['vadJsSource'] = vadJsSource.toJS;
    initMsg['asrJsSource'] = asrJsSource.toJS;
    initMsg['wasmBinary'] = wasmData.buffer.toJS;
    initMsg['vadModelFiles'] = jsVadModelFiles;
    initMsg['asrModelFiles'] = jsAsrModelFiles;
    initMsg['vadConfig'] = m.configToJs(vadConfig);
    initMsg['asrConfig'] = asrConfig;
    _worker!.postMessage(initMsg);
  }

  /// Run VAD+ASR on audio samples.
  void runVad({
    required int runId,
    required Float32List samples,
    required int sampleRate,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) {
    final msg = JSObject();
    _currentRunId = runId;
    msg['type'] = 'runVad'.toJS;
    msg['runId'] = runId.toJS;
    msg['samples'] = samples.buffer.toJS;
    msg['sampleRate'] = sampleRate.toJS;
    msg['threshold'] = threshold.toJS;
    msg['minSilenceDuration'] = minSilenceDuration.toJS;
    msg['minSpeechDuration'] = minSpeechDuration.toJS;
    msg['maxSpeechDuration'] = maxSpeechDuration.toJS;
    _worker?.postMessage(msg);
  }

  void cancel() {
    final msg = JSObject();
    msg['type'] = 'cancel'.toJS;
    _worker?.postMessage(msg);
  }

  void dispose() {
    _worker?.terminate();
    _worker = null;
  }

  static Future<String> _loadAssetAsString(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return utf8.decode(
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes));
  }

  static Future<Uint8List> _loadAssetBytes(String assetPath) async {
    final data = await rootBundle.load(assetPath);
    return data.buffer
        .asUint8List(data.offsetInBytes, data.lengthInBytes);
  }

  /// Build ASR config as a JS object for the worker.
  JSObject _buildAsrConfigForWeb() {
    final jsConfig = JSObject();

    // Feature config.
    final feat = JSObject();
    feat['sampleRate'] = 16000.toJS;
    feat['featureDim'] = 80.toJS;
    jsConfig['featConfig'] = feat;

    // Model config — only set the active model's fields.
    final modelConfig = JSObject();
    final modelDir = cfg.selectedAsrModel.assetFiles[0]
        .substring(0, cfg.selectedAsrModel.assetFiles[0].lastIndexOf('/'));

    // IMPORTANT: Keep this switch in sync with model_config.dart's
    // buildAsrConfig() and the asrModels list.
    switch (cfg.selectedModelIndex) {
      case 0: // Zipformer CTC
        final zipformerCtc = JSObject();
        zipformerCtc['model'] = '$modelDir/model.int8.onnx'.toJS;
        modelConfig['zipformerCtc'] = zipformerCtc;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        modelConfig['modelingUnit'] = 'cjkchar'.toJS;
        break;
      case 1: // SenseVoice
        final senseVoice = JSObject();
        senseVoice['model'] = '$modelDir/model.int8.onnx'.toJS;
        senseVoice['language'] = 'auto'.toJS;
        modelConfig['senseVoice'] = senseVoice;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        break;
      case 2: // Whisper
        final whisper = JSObject();
        whisper['encoder'] = '$modelDir/tiny.en-encoder.int8.onnx'.toJS;
        whisper['decoder'] = '$modelDir/tiny.en-decoder.int8.onnx'.toJS;
        modelConfig['whisper'] = whisper;
        modelConfig['tokens'] = '$modelDir/tiny.en-tokens.txt'.toJS;
        modelConfig['modelType'] = 'whisper'.toJS;
        break;
      case 3: // NeMo Parakeet (English, transducer)
      case 4: // NeMo Parakeet TDT v3 (transducer)
        final transducer = JSObject();
        transducer['encoder'] = '$modelDir/encoder.int8.onnx'.toJS;
        transducer['decoder'] = '$modelDir/decoder.int8.onnx'.toJS;
        transducer['joiner'] = '$modelDir/joiner.int8.onnx'.toJS;
        modelConfig['transducer'] = transducer;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        modelConfig['modelType'] = 'nemo_transducer'.toJS;
        break;
      case 5: // Moonshine tiny en
        final moonshine = JSObject();
        moonshine['encoder'] = '$modelDir/encoder_model.ort'.toJS;
        moonshine['mergedDecoder'] = '$modelDir/decoder_model_merged.ort'.toJS;
        modelConfig['moonshine'] = moonshine;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        break;
      case 6: // Qwen3 ASR
        final qwen3Asr = JSObject();
        qwen3Asr['convFrontend'] = '$modelDir/conv_frontend.onnx'.toJS;
        qwen3Asr['encoder'] = '$modelDir/encoder.int8.onnx'.toJS;
        qwen3Asr['decoder'] = '$modelDir/decoder.int8.onnx'.toJS;
        qwen3Asr['tokenizer'] = '$modelDir/tokenizer/tokenizer.json'.toJS;
        modelConfig['qwen3Asr'] = qwen3Asr;
        break;
      case 7: // FunASR Nano
        final funasrNano = JSObject();
        funasrNano['encoderAdaptor'] = '$modelDir/encoder_adaptor.int8.onnx'.toJS;
        funasrNano['llm'] = '$modelDir/llm.int8.onnx'.toJS;
        funasrNano['embedding'] = '$modelDir/embedding.int8.onnx'.toJS;
        funasrNano['tokenizer'] = '$modelDir/Qwen3-0.6B/tokenizer.json'.toJS;
        modelConfig['funasrNano'] = funasrNano;
        break;
      case 8: // FireRed ASR CTC
        final fireRedAsrCtc = JSObject();
        fireRedAsrCtc['model'] = '$modelDir/model.int8.onnx'.toJS;
        modelConfig['fireRedAsrCtc'] = fireRedAsrCtc;
        modelConfig['tokens'] = '$modelDir/tokens.txt'.toJS;
        break;
    }

    modelConfig['numThreads'] = 1.toJS;
    modelConfig['debug'] = 1.toJS;
    modelConfig['provider'] = 'cpu'.toJS;

    jsConfig['modelConfig'] = modelConfig;

    // All other fields (LM, HR, decoding params) use JS defaults.
    return jsConfig;
  }

  void _handleMessage(web.MessageEvent event) {
    final data = event.data! as JSObject;
    final type = (data.getProperty('type'.toJS)! as JSString).toDart;

    if (type == 'ready') {
      onReady();
    } else if (type == 'started') {
      final runId = (data.getProperty('runId'.toJS) as JSNumber?)?.toDartInt ?? 0;
      onStarted(runId);
    } else if (type == 'segment') {
      // A single segment with ASR text — display it immediately.
      try {
        final segRunId =
            (data.getProperty('runId'.toJS) as JSNumber?)?.toDartInt ?? 0;
        if (segRunId != _currentRunId) return; // Stale segment from old run.
        final segData = data;
        final start =
            (segData.getProperty('start'.toJS)! as JSNumber).toDartDouble;
        final end =
            (segData.getProperty('end'.toJS)! as JSNumber).toDartDouble;
        final text =
            (segData.getProperty('text'.toJS) as JSString?)?.toDart ?? '';
        final samplesRaw = segData.getProperty('samples'.toJS)!;
        Float32List samples;
        if (samplesRaw.isA<JSFloat32Array>()) {
          samples =
              Float32List.fromList((samplesRaw as JSFloat32Array).toDart);
        } else {
          final arr = samplesRaw as JSArray;
          final list = List<double>.filled(arr.length, 0);
          for (int j = 0; j < arr.length; j++) {
            list[j] = (arr[j] as JSNumber).toDartDouble;
          }
          samples = Float32List.fromList(list);
        }
        onSegment(VadAsrSegment(
          start: start,
          end: end,
          samples: samples,
          text: text,
        ));
      } catch (e) {
        if (kDebugMode) print('[worker_web] segment parse error: $e');
      }
    } else if (type == 'progress') {
      final progressRunId =
          (data.getProperty('runId'.toJS) as JSNumber?)?.toDartInt ?? 0;
      if (progressRunId == _currentRunId) {
        final progress =
            (data.getProperty('progress'.toJS)! as JSNumber).toDartDouble;
        onProgress(progress);
      }
    } else if (type == 'result') {
      try {
        final resultRunId =
            (data.getProperty('runId'.toJS) as JSNumber?)?.toDartInt ?? 0;
        if (resultRunId != _currentRunId) return;
        final segmentsJs = data.getProperty('segments'.toJS)! as JSArray;
        final elapsed =
            (data.getProperty('elapsed'.toJS)! as JSNumber).toDartDouble;
        final audioDuration =
            (data.getProperty('audioDuration'.toJS)! as JSNumber).toDartDouble;

        final segments = <VadAsrSegment>[];
        for (int i = 0; i < segmentsJs.length; i++) {
          final seg = segmentsJs[i] as JSObject;
          final start =
              (seg.getProperty('start'.toJS)! as JSNumber).toDartDouble;
          final end =
              (seg.getProperty('end'.toJS)! as JSNumber).toDartDouble;
          final text =
              (seg.getProperty('text'.toJS) as JSString?)?.toDart ?? '';

          // Samples may be Float32Array or regular Array after transfer.
          final samplesRaw = seg.getProperty('samples'.toJS)!;
          Float32List samples;
          if (samplesRaw.isA<JSFloat32Array>()) {
            samples = Float32List.fromList(
                (samplesRaw as JSFloat32Array).toDart);
          } else {
            // Fallback: iterate as JSArray of numbers.
            final arr = samplesRaw as JSArray;
            final list = List<double>.filled(arr.length, 0);
            for (int j = 0; j < arr.length; j++) {
              list[j] = (arr[j] as JSNumber).toDartDouble;
            }
            samples = Float32List.fromList(list);
          }

          segments.add(VadAsrSegment(
            start: start,
            end: end,
            samples: samples,
            text: text,
          ));
        }
        onResult(segments, elapsed, audioDuration);
      } catch (e) {
        onError('Failed to parse result: $e');
      }
    } else if (type == 'log') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      if (kDebugMode) print('[vad-asr-worker] $msg');
    } else if (type == 'error') {
      final msg =
          (data.getProperty('message'.toJS)! as JSString).toDart;
      onError(msg);
    }
  }
}
