// Copyright (c)  2024  Xiaomi Corporation

import "dart:io";

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';

Future<sherpa_onnx.OfflineTts> createOfflineTts() async {
  // sherpa_onnx requires that model files are in the local disk, so we
  // need to copy all asset files to disk.
  await copyAllAssetFiles();

  sherpa_onnx.initBindings();

  // Such a design is to make it easier to build flutter APPs with
  // github actions for a variety of tts models
  //
  // See https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/flutter/generate-tts.py
  // for details

  String modelDir = '';
  String modelName = '';
  String ruleFsts = '';
  String ruleFars = '';
  String lexicon = '';
  String dataDir = '';
  String dictDir = '';

  // You can select an example below and change it according to match your
  // selected tts model

  // ============================================================
  // Your change starts here
  // ============================================================

  // Example 1:
  // modelDir = 'vits-vctk';
  // modelName = 'vits-vctk.onnx';
  // lexicon = 'lexicon.txt';

  // Example 2:
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
  // modelDir = 'vits-piper-en_US-amy-low';
  // modelName = 'en_US-amy-low.onnx';
  // dataDir = 'vits-piper-en_US-amy-low/espeak-ng-data';

  // Example 3:
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
  // modelDir = 'vits-icefall-zh-aishell3';
  // modelName = 'model.onnx';
  // ruleFsts = 'vits-icefall-zh-aishell3/phone.fst,vits-icefall-zh-aishell3/date.fst,vits-icefall-zh-aishell3/number.fst,vits-icefall-zh-aishell3/new_heteronym.fst';
  // ruleFars = 'vits-icefall-zh-aishell3/rule.far';
  // lexicon = 'lexicon.txt';

  // Example 4:
  // https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/vits.html#csukuangfj-vits-zh-hf-fanchen-c-chinese-187-speakers
  // modelDir = 'vits-zh-hf-fanchen-C';
  // modelName = 'vits-zh-hf-fanchen-C.onnx';
  // lexicon = 'lexicon.txt';
  // dictDir = 'vits-zh-hf-fanchen-C/dict';

  // Example 5:
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
  // modelDir = 'vits-coqui-de-css10';
  // modelName = 'model.onnx';

  // Example 6
  // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-libritts_r-medium.tar.bz2
  // modelDir = 'vits-piper-en_US-libritts_r-medium';
  // modelName = 'en_US-libritts_r-medium.onnx';
  // dataDir = 'vits-piper-en_US-libritts_r-medium/espeak-ng-data';

  // ============================================================
  // Please don't change the remaining part of this function
  // ============================================================
  if (modelName == '') {
    throw Exception(
        'You are supposed to select a model by changing the code before you run the app');
  }

  final Directory directory = await getApplicationDocumentsDirectory();
  modelName = p.join(directory.path, modelDir, modelName);

  if (ruleFsts != '') {
    final all = ruleFsts.split(',');
    var tmp = <String>[];
    for (final f in all) {
      tmp.add(p.join(directory.path, f));
    }
    ruleFsts = tmp.join(',');
  }

  if (ruleFars != '') {
    final all = ruleFars.split(',');
    var tmp = <String>[];
    for (final f in all) {
      tmp.add(p.join(directory.path, f));
    }
    ruleFars = tmp.join(',');
  }

  if (lexicon != '') {
    lexicon = p.join(directory.path, modelDir, lexicon);
  }

  if (dataDir != '') {
    dataDir = p.join(directory.path, dataDir);
  }

  if (dictDir != '') {
    dictDir = p.join(directory.path, dictDir);
  }

  final tokens = p.join(directory.path, modelDir, 'tokens.txt');

  final vits = sherpa_onnx.OfflineTtsVitsModelConfig(
    model: modelName,
    lexicon: lexicon,
    tokens: tokens,
    dataDir: dataDir,
    dictDir: dictDir,
  );

  final modelConfig = sherpa_onnx.OfflineTtsModelConfig(
    vits: vits,
    numThreads: 2,
    debug: true,
    provider: 'cpu',
  );

  final config = sherpa_onnx.OfflineTtsConfig(
    model: modelConfig,
    ruleFsts: ruleFsts,
    ruleFars: ruleFars,
    maxNumSenetences: 1,
  );
  // print(config);

  final tts = sherpa_onnx.OfflineTts(config);
  print('tts created successfully');

  return tts;
}
