import 'dart:io';
import 'dart:isolate';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:media_kit/media_kit.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import 'utils.dart';

class _IsolateTask<T> {
  final SendPort sendPort;

  RootIsolateToken? rootIsolateToken;

  _IsolateTask(this.sendPort, this.rootIsolateToken);
}

class _PortModel {
  final String method;

  final SendPort? sendPort;
  dynamic data;

  _PortModel({
    required this.method,
    this.sendPort,
    this.data,
  });
}

class _TtsManager {
  /// 主进程通信端口
  final ReceivePort receivePort;

  final Isolate isolate;

  final SendPort isolatePort;

  _TtsManager({
    required this.receivePort,
    required this.isolate,
    required this.isolatePort,
  });
}

class IsolateTts {
  static late final _TtsManager _ttsManager;

  /// 获取线程里的通信端口
  static SendPort get _sendPort => _ttsManager.isolatePort;

  static late sherpa_onnx.OfflineTts _tts;

  static late Player _player;

  static Future<void> init() async {
    ReceivePort port = ReceivePort();
    RootIsolateToken? rootIsolateToken = RootIsolateToken.instance;

    Isolate isolate = await Isolate.spawn(
      _isolateEntry,
      _IsolateTask(port.sendPort, rootIsolateToken),
      errorsAreFatal: false,
    );
    port.listen((msg) async {
      if (msg is SendPort) {
        print(11);
        _ttsManager = _TtsManager(receivePort: port, isolate: isolate, isolatePort: msg);
        return;
      }
    });
  }

  static Future<void> _isolateEntry(_IsolateTask task) async {
    if (task.rootIsolateToken != null) {
      BackgroundIsolateBinaryMessenger.ensureInitialized(task.rootIsolateToken!);
    }
    MediaKit.ensureInitialized();
    _player = Player();
    sherpa_onnx.initBindings();
    final receivePort = ReceivePort();
    task.sendPort.send(receivePort.sendPort);

    String modelDir = '';
    String modelName = '';
    String ruleFsts = '';
    String ruleFars = '';
    String lexicon = '';
    String dataDir = '';
    String dictDir = '';

    // Example 7
    // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
    // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-melo-tts-zh_en.tar.bz2
    modelDir = 'vits-melo-tts-zh_en';
    modelName = 'model.onnx';
    lexicon = 'lexicon.txt';
    dictDir = 'vits-melo-tts-zh_en/dict';

    if (modelName == '') {
      throw Exception('You are supposed to select a model by changing the code before you run the app');
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
    receivePort.listen((msg) async {
      print(msg);
      if (msg is _PortModel) {
        switch (msg.method) {
          case 'generate':
            {
              _PortModel _v = msg;
              final stopwatch = Stopwatch();
              stopwatch.start();
              final audio = _tts.generate(text: _v.data['text'], sid: _v.data['sid'], speed: _v.data['speed']);
              final suffix = '-sid-${_v.data['sid']}-speed-${_v.data['sid'].toStringAsPrecision(2)}';
              final filename = await generateWaveFilename(suffix);

              final ok = sherpa_onnx.writeWave(
                filename: filename,
                samples: audio.samples,
                sampleRate: audio.sampleRate,
              );

              if (ok) {
                stopwatch.stop();
                double elapsed = stopwatch.elapsed.inMilliseconds.toDouble();

                double waveDuration = audio.samples.length.toDouble() / audio.sampleRate.toDouble();

                print('Saved to\n$filename\n'
                    'Elapsed: ${(elapsed / 1000).toStringAsPrecision(4)} s\n'
                    'Wave duration: ${waveDuration.toStringAsPrecision(4)} s\n'
                    'RTF: ${(elapsed / 1000).toStringAsPrecision(4)}/${waveDuration.toStringAsPrecision(4)} '
                    '= ${(elapsed / 1000 / waveDuration).toStringAsPrecision(3)} ');

                await _player.open(Media('file:///$filename'));
                await _player.play();
              }
            }
            break;
        }
      }
    });
    _tts = sherpa_onnx.OfflineTts(config);
  }

  static Future<void> generate({required String text, int sid = 0, double speed = 1.0}) async {
    ReceivePort receivePort = ReceivePort();
    _sendPort.send(_PortModel(
      method: 'generate',
      data: {'text': text, 'sid': sid, 'speed': speed},
      sendPort: receivePort.sendPort,
    ));
    await receivePort.first;
    receivePort.close();
  }
}

/// 这里是页面
class IsolateTtsView extends StatefulWidget {
  const IsolateTtsView({super.key});

  @override
  State<IsolateTtsView> createState() => _IsolateTtsViewState();
}

class _IsolateTtsViewState extends State<IsolateTtsView> {
  @override
  void initState() {
    super.initState();
    IsolateTts.init();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: ElevatedButton(
          onPressed: () {
            IsolateTts.generate(text: '这是已退出的 isolate TTS');
          },
          child: Text('Isolate TTS'),
        ),
      ),
    );
  }
}
