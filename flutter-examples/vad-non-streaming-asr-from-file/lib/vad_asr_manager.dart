// Copyright (c)  2026  Xiaomi Corporation
//
// VAD + Non-streaming ASR manager (native only, uses isolates).
// Single isolate runs both VAD and ASR: each VAD segment is decoded by ASR
// immediately, matching the web worker's pipeline behavior.
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './model.dart' as model;
import './model_config.dart' as cfg;

/// A VAD segment with ASR transcription.
class VadAsrSegment {
  final double start;
  final double end;
  final Float32List samples;
  String text;

  VadAsrSegment({
    required this.start,
    required this.end,
    required this.samples,
    this.text = '',
  });
}

class VadAsrResult {
  final List<VadAsrSegment> segments;
  final double elapsed;
  final double audioDuration;
  const VadAsrResult({
    required this.segments,
    required this.elapsed,
    required this.audioDuration,
  });
}

enum VadAsrState { uninitialized, initializing, initialized }

class VadAsrManager {
  VadAsrState _state = VadAsrState.uninitialized;
  VadAsrState get state => _state;

  final _logController = StreamController<String>.broadcast();
  final _progressController = StreamController<double>.broadcast();
  final _segmentController = StreamController<VadAsrSegment>.broadcast();
  final _resultController = StreamController<VadAsrResult>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<double> get progressStream => _progressController.stream;
  Stream<VadAsrSegment> get segmentStream => _segmentController.stream;
  Stream<VadAsrResult> get resultStream => _resultController.stream;

  Isolate? _isolate;
  SendPort? _sendPort;
  ReceivePort? _receivePort;

  final List<VadAsrSegment> _segments = [];

  Future<void> init({required String modelDir, required String vadModelDir}) async {
    if (_state != VadAsrState.uninitialized) return;
    _state = VadAsrState.initializing;
    _logController.add('Initializing VAD + ASR...');

    try {
      _receivePort = ReceivePort();

      // Get VAD config.
      await model.prepareModelConfig();
      final baseVadConfig = cfg.defaultVadConfig;
      final vadConfig = sherpa_onnx.VadModelConfig(
        sileroVad: sherpa_onnx.SileroVadModelConfig(
          model: '$vadModelDir/${baseVadConfig.sileroVad.model}',
          threshold: baseVadConfig.sileroVad.threshold,
          minSilenceDuration: baseVadConfig.sileroVad.minSilenceDuration,
          minSpeechDuration: baseVadConfig.sileroVad.minSpeechDuration,
          windowSize: baseVadConfig.sileroVad.windowSize,
          maxSpeechDuration: baseVadConfig.sileroVad.maxSpeechDuration,
        ),
        numThreads: baseVadConfig.numThreads,
        debug: baseVadConfig.debug,
      );

      // Spawn single isolate for both VAD and ASR.
      _isolate = await Isolate.spawn(
        _workerEntry,
        _InitRequest(
          vadConfig: vadConfig,
          asrModelDir: modelDir,
          mainSendPort: _receivePort!.sendPort,
        ),
      );

      // Wait for isolate to be ready.
      final readyCompleter = Completer<void>();
      bool initDone = false;

      _receivePort!.listen((message) {
        if (!initDone) {
          initDone = true;
          if (message is _IsolateReady && message.isSuccess) {
            _sendPort = message.sendPort!;
            readyCompleter.complete();
          } else if (message is _IsolateReady) {
            readyCompleter.completeError(Exception('Init failed: ${message.error}'));
          } else {
            readyCompleter.completeError(Exception('Unexpected first message: $message'));
          }
        } else {
          _onWorkerMessage(message);
        }
      });

      await readyCompleter.future.timeout(
        const Duration(seconds: 30),
        onTimeout: () => throw TimeoutException('Isolate init timed out'),
      );

      _state = VadAsrState.initialized;
      _logController.add('Ready (VAD + ASR initialized)');
    } catch (e) {
      _state = VadAsrState.uninitialized;
      _logController.add('Init error: $e');
      rethrow;
    }
  }

  void _onWorkerMessage(dynamic message) {
    if (message is _ProgressUpdate) {
      _progressController.add(message.progress);
    } else if (message is _SegmentFound) {
      if (message.text.trim().isEmpty) return;
      final seg = VadAsrSegment(
        start: message.start,
        end: message.end,
        samples: message.samples,
        text: message.text,
      );
      _segments.add(seg);
      _segmentController.add(seg);
    } else if (message is _RunDone) {
      _resultController.add(VadAsrResult(
        segments: List.unmodifiable(_segments),
        elapsed: message.elapsed,
        audioDuration: message.audioDuration,
      ));
    }
  }

  void runVad({
    required Float32List samples,
    required int sampleRate,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) {
    if (_state != VadAsrState.initialized) {
      _logController.add('Error: not initialized');
      return;
    }
    _segments.clear();
    _sendPort!.send(_RunRequest(
      samples: samples,
      sampleRate: sampleRate,
      threshold: threshold,
      minSilenceDuration: minSilenceDuration,
      minSpeechDuration: minSpeechDuration,
      maxSpeechDuration: maxSpeechDuration,
    ));
  }

  void cancel() {
    _isolate?.kill(priority: Isolate.immediate);
    _isolate = null;
    _sendPort = null;
    _receivePort?.close();
    _receivePort = null;
    _state = VadAsrState.uninitialized;
  }

  void dispose() {
    cancel();
    _logController.close();
    _progressController.close();
    _segmentController.close();
    _resultController.close();
  }
}

// ── Messages ──────────────────────────────────────────────────────────────

class _RunRequest {
  final Float32List samples;
  final int sampleRate;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final double maxSpeechDuration;
  _RunRequest({
    required this.samples,
    required this.sampleRate,
    required this.threshold,
    required this.minSilenceDuration,
    required this.minSpeechDuration,
    required this.maxSpeechDuration,
  });
}

class _IsolateReady {
  final SendPort? sendPort;
  final String? error;
  _IsolateReady.sendPort(this.sendPort) : error = null;
  _IsolateReady.error(this.error) : sendPort = null;
  bool get isSuccess => sendPort != null;
}

class _ProgressUpdate {
  final double progress;
  _ProgressUpdate(this.progress);
}

class _SegmentFound {
  final double start;
  final double end;
  final Float32List samples;
  final String text;
  _SegmentFound(this.start, this.end, this.samples, this.text);
}

class _RunDone {
  final double elapsed;
  final double audioDuration;
  _RunDone(this.elapsed, this.audioDuration);
}

// ── Worker Isolate (VAD + ASR in one isolate) ─────────────────────────────

class _InitRequest {
  final sherpa_onnx.VadModelConfig vadConfig;
  final String asrModelDir;
  final SendPort mainSendPort;
  _InitRequest({
    required this.vadConfig,
    required this.asrModelDir,
    required this.mainSendPort,
  });
}

void _workerEntry(_InitRequest initReq) {
  try {
    sherpa_onnx.initBindings();

    // Create ASR recognizer once.
    final asrConfig = cfg.buildAsrConfig(modelDir: initReq.asrModelDir);
    final recognizer = sherpa_onnx.OfflineRecognizer(asrConfig);

    final receivePort = ReceivePort();
    initReq.mainSendPort.send(_IsolateReady.sendPort(receivePort.sendPort));

    sherpa_onnx.VoiceActivityDetector? vad;
    final watch = Stopwatch();

    receivePort.listen((message) {
      if (message is _RunRequest) {
        // Recreate VAD with user parameters.
        vad?.free();
        final baseCfg = initReq.vadConfig;
        vad = sherpa_onnx.VoiceActivityDetector(
          config: sherpa_onnx.VadModelConfig(
            sileroVad: sherpa_onnx.SileroVadModelConfig(
              model: baseCfg.sileroVad.model,
              threshold: message.threshold,
              minSilenceDuration: message.minSilenceDuration,
              minSpeechDuration: message.minSpeechDuration,
              windowSize: baseCfg.sileroVad.windowSize,
              maxSpeechDuration: message.maxSpeechDuration,
            ),
            numThreads: baseCfg.numThreads,
            debug: baseCfg.debug,
          ),
          bufferSizeInSeconds: 300,
        );

        watch.reset();
        watch.start();

        final windowSize = baseCfg.sileroVad.windowSize;
        final numSamples = message.samples.length;
        final numIter = numSamples ~/ windowSize;

        // Pipeline: VAD detects → ASR decodes immediately → send result.
        void processSegments() {
          while (!vad!.isEmpty()) {
            final segSamples = vad!.front().samples;
            final startSec = vad!.front().start / message.sampleRate;
            final endSec = startSec + segSamples.length / message.sampleRate;

            // Decode this segment immediately.
            final stream = recognizer.createStream();
            stream.acceptWaveform(
                samples: segSamples, sampleRate: message.sampleRate);
            recognizer.decode(stream);
            final result = recognizer.getResult(stream);
            stream.free();

            initReq.mainSendPort.send(
                _SegmentFound(startSec, endSec, segSamples, result.text));

            vad!.pop();
          }
        }

        for (int i = 0; i != numIter; ++i) {
          final start = i * windowSize;
          vad!.acceptWaveform(
              Float32List.sublistView(message.samples, start, start + windowSize));
          processSegments();
          initReq.mainSendPort.send(_ProgressUpdate((i + 1) / numIter));
        }

        vad!.flush();
        processSegments();

        watch.stop();
        initReq.mainSendPort.send(_ProgressUpdate(1.0));
        initReq.mainSendPort.send(_RunDone(
            watch.elapsedMilliseconds / 1000.0, numSamples / message.sampleRate));
      }
    });
  } catch (e) {
    initReq.mainSendPort.send(_IsolateReady.error('$e'));
  }
}
