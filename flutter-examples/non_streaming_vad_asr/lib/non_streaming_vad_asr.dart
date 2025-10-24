// Copyright (c)  2024  Xiaomi Corporation
import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';
import './offline_model.dart';

final modelDir = 'assets';
Future<sherpa_onnx.OfflineRecognizer> createOfflineRecognizer() async {
  final type = 2;
  final modelConfig = await getOfflineModelConfig(type: type);
  final config = sherpa_onnx.OfflineRecognizerConfig(model: modelConfig);
  return sherpa_onnx.OfflineRecognizer(config);
}

class NoStreamingAsrVAdScreen extends StatefulWidget {
  const NoStreamingAsrVAdScreen({super.key});

  @override
  State<NoStreamingAsrVAdScreen> createState() => _NoStreamingAsrVAdScreenState();
}

class _NoStreamingAsrVAdScreenState extends State<NoStreamingAsrVAdScreen> {

  late final TextEditingController _controller;
  late final AudioRecorder _audioRecorder;

  String _title = 'Real-time speech recognition(offline recognizer with vad)';
  String _last = '';
  int _index = 0;
  bool _isInitialized = false;

  // offline recognizer related vars
  sherpa_onnx.OfflineRecognizer? _recognizer;
  static const int _sampleRate = 16000;

  // VAD related vars
  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.CircularBuffer? _buffer;
  
  // VAD config
  late sherpa_onnx.VadModelConfig _vadConfig;

  StreamSubscription<RecordState>? _recordSub;
  RecordState _recordState = RecordState.stop;

  @override
  void initState() {
    _audioRecorder = AudioRecorder();
    _controller = TextEditingController();

    _recordSub = _audioRecorder.onStateChanged().listen((recordState) {
      _updateRecordState(recordState);
    });

    super.initState();
  }

  Future<void> _start() async {
    if (!_isInitialized) {
      sherpa_onnx.initBindings();

      // 初始化 VAD
      final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
        model: await copyAssetFile('$modelDir/silero_vad.onnx'),
        minSilenceDuration: 0.25,
        minSpeechDuration: 0.5,
        maxSpeechDuration: 5.0,
      );

      _vadConfig = sherpa_onnx.VadModelConfig(
        sileroVad: sileroVadConfig,
        numThreads: 1,
        debug: false,
      );

      // create VAD, use buffer model
      _vad = sherpa_onnx.VoiceActivityDetector(
        config: _vadConfig, 
        bufferSizeInSeconds: 30
      );
      _buffer = sherpa_onnx.CircularBuffer(capacity: 30 * _sampleRate);

      _recognizer = await createOfflineRecognizer();
      _isInitialized = true;
    }

    try {
      if (await _audioRecorder.hasPermission()) {
        const encoder = AudioEncoder.pcm16bits;

        if (!await _isEncoderSupported(encoder)) {
          return;
        }

        final devs = await _audioRecorder.listInputDevices();
        debugPrint(devs.toString());

        const config = RecordConfig(
          encoder: encoder,
          sampleRate: _sampleRate,
          numChannels: 1,
        );

        final stream = await _audioRecorder.startStream(config);

        stream.listen(
          (data) {
            final samplesFloat32 = convertBytesToFloat32(Uint8List.fromList(data));
            
            // use _buffer and _vad for offline stream data making
            _buffer!.push(samplesFloat32);
            
            final windowSize = _vadConfig.sileroVad.windowSize;
            while (_buffer!.size > windowSize) {
              final samples = _buffer!.get(
                startIndex: _buffer!.head, 
                n: windowSize
              );
              _buffer!.pop(windowSize);
              _vad!.acceptWaveform(samples);  

              while (!_vad!.isEmpty()) {
                final segment = _vad!.front();
                final samples = segment.samples;  
                
                // offline _recognizer stream handle logic
                final stream = _recognizer!.createStream();
                stream.acceptWaveform(samples: samples, sampleRate: _sampleRate);
                _recognizer!.decode(stream);
                final text = _recognizer!.getResult(stream).text;
                debugPrint("recognize:"+text);
                stream.free();
                _vad!.pop();
                
                // update text to display
                String textToDisplay = _last;
                if (text != '') {
                  _index += 1;
                  if (_last == '') {
                    textToDisplay = '$_index: $text';
                  } else {
                    textToDisplay = '$_index: $text\n$_last';
                  }
                  _last = textToDisplay;
                }
                debugPrint("final:"+textToDisplay);
                _controller.value = TextEditingValue(
                      text: textToDisplay,
                      selection: TextSelection.collapsed(offset: textToDisplay.length),
                );
              }
            }
          },
          onDone: () {
            print('stream stopped.');
          },
        );
      }
    } catch (e) {
      print(e);
    }
  }

  Future<void> _stop() async {
    await _audioRecorder.stop();
    // handle rest of vad data
     _vad!.flush();
    while (!_vad!.isEmpty()) {
      final segment = _vad!.front();
      final samples = segment.samples;

      final stream = _recognizer!.createStream();
      stream.acceptWaveform(samples: samples, sampleRate: _sampleRate);
      _recognizer!.decode(stream);
      final text = _recognizer!.getResult(stream).text;
              
      String textToDisplay = _last;
      if (text != '') {
          _index += 1;
          if (_last == '') {
              textToDisplay = '$_index: $text';
          } else {
              textToDisplay = '$_index: $text\n$_last';
            }
          }
          _last = "";
          _index = 0;  
          debugPrint("final:"+textToDisplay);
          _controller.value = TextEditingValue(
            text: textToDisplay,
            selection: TextSelection.collapsed(offset: textToDisplay.length),
          );              
          stream.free();
          _vad!.pop();
    }
  }

  Future<void> _pause() => _audioRecorder.pause();

  Future<void> _resume() => _audioRecorder.resume();

  void _updateRecordState(RecordState recordState) {
    setState(() => _recordState = recordState);
  }

  Future<bool> _isEncoderSupported(AudioEncoder encoder) async {
    final isSupported = await _audioRecorder.isEncoderSupported(
      encoder,
    );

    if (!isSupported) {
      debugPrint('${encoder.name} is not supported on this platform.');
      debugPrint('Supported encoders are:');

      for (final e in AudioEncoder.values) {
        if (await _audioRecorder.isEncoderSupported(e)) {
          debugPrint('- ${e.name}');
        }
      }
    }

    return isSupported;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        appBar: AppBar(
          title: Text(_title),
        ),
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(height: 50),
            TextField(
              maxLines: 5,
              controller: _controller,
              readOnly: true,
            ),
            const SizedBox(height: 50),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                _buildRecordStopControl(),
                const SizedBox(width: 20),
                _buildText(),
              ],
            ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _recordSub?.cancel();
    _audioRecorder.dispose();
    _recognizer?.free();
    _vad?.free(); // release vad
    _buffer?.free(); // release buffer
    super.dispose();
  }

  Widget _buildRecordStopControl() {
    late Icon icon;
    late Color color;

    if (_recordState != RecordState.stop) {
      icon = const Icon(Icons.stop, color: Colors.red, size: 30);
      color = Colors.red.withOpacity(0.1);
    } else {
      final theme = Theme.of(context);
      icon = Icon(Icons.mic, color: theme.primaryColor, size: 30);
      color = theme.primaryColor.withOpacity(0.1);
    }

    return ClipOval(
      child: Material(
        color: color,
        child: InkWell(
          child: SizedBox(width: 56, height: 56, child: icon),
          onTap: () {
            (_recordState != RecordState.stop) ? _stop() : _start();
          },
        ),
      ),
    );
  }

  Widget _buildText() {
    if (_recordState == RecordState.stop) {
      return const Text("Start");
    } else {
      return const Text("Stop");
    }
  }
}