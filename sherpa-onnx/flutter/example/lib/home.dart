// Copyright (c)  2024  Xiaomi Corporation
import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './utils.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  late final AudioRecorder _audioRecorder;

  bool _printed = false;
  var _color = Colors.black;
  bool _isInitialized = false;

  sherpa_onnx.VoiceActivityDetector? _vad;
  sherpa_onnx.CircularBuffer? _buffer;

  StreamSubscription<RecordState>? _recordSub;
  RecordState _recordState = RecordState.stop;

  @override
  void initState() {
    _audioRecorder = AudioRecorder();

    _recordSub = _audioRecorder.onStateChanged().listen((recordState) {
      _updateRecordState(recordState);
    });

    super.initState();
  }

  Future<void> _start() async {
    if (!_isInitialized) {
      sherpa_onnx.initBindings();
      final src = 'assets/silero_vad.onnx';
      final modelPath = await copyAssetFile(src: src, dst: 'silero_vad.onnx');

      final sileroVadConfig = sherpa_onnx.SileroVadModelConfig(
        model: modelPath,
        minSpeechDuration: 0.25,
        minSilenceDuration: 0.5,
      );

      final config = sherpa_onnx.VadModelConfig(
        sileroVad: sileroVadConfig,
        numThreads: 1,
        debug: true,
      );

      _vad = sherpa_onnx.VoiceActivityDetector(
          config: config, bufferSizeInSeconds: 30);

      _buffer = sherpa_onnx.CircularBuffer(capacity: 16000 * 30);
      print(_buffer!.ptr);

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
          sampleRate: 16000,
          numChannels: 1,
        );

        final stream = await _audioRecorder.startStream(config);

        final dir = await getApplicationDocumentsDirectory();

        stream.listen(
          (data) {
            final samplesFloat32 =
                convertBytesToFloat32(Uint8List.fromList(data));

            _buffer!.push(samplesFloat32);

            final windowSize = _vad!.config.sileroVad.windowSize;
            while (_buffer!.size > windowSize) {
              final samples =
                  _buffer!.get(startIndex: _buffer!.head, n: windowSize);
              _buffer!.pop(windowSize);
              _vad!.acceptWaveform(samples);
              if (_vad!.isDetected() && !_printed) {
                print('detected');
                _printed = true;

                setState(() => _color = Colors.red);
              }

              if (!_vad!.isDetected()) {
                _printed = false;
                setState(() => _color = Colors.black);
              }

              while (!_vad!.isEmpty()) {
                final segment = _vad!.front();
                final duration = segment.samples.length / 16000;
                final d = DateTime.now();
                final filename = p.join(dir.path,
                    '${d.year}-${d.month}-${d.day}-${d.hour}-${d.minute}-${d.second}-duration-${duration.toStringAsPrecision(3)}s.wav');

                bool ok = sherpa_onnx.writeWave(
                    filename: filename,
                    samples: segment.samples,
                    sampleRate: 16000);
                if (!ok) {
                  print('Failed to write $filename');
                } else {
                  print('Saved to write $filename');
                }

                _vad!.pop();
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
    _buffer!.reset();
    _vad!.clear();

    await _audioRecorder.stop();
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
          debugPrint('- ${encoder.name}');
        }
      }
    }

    return isSupported;
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      home: Scaffold(
        body: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Container(
              width: 100.0,
              height: 100.0,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _color,
              ),
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
    _vad?.free();
    _buffer?.free();
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
