// Copyright (c)  2026  Xiaomi Corporation
//
// VAD + Non-streaming ASR manager for microphone input.
// Two background isolates:
//   1. VAD isolate — runs VoiceActivityDetector
//   2. ASR isolate — runs OfflineRecognizer
//
// ## Message flow
//
// ### Init
//   UI spawns ASR isolate → ASR sends its SendPort
//   UI spawns VAD isolate (with ASR SendPort) → VAD sends its SendPort
//
// ### Run (real-time microphone)
//   UI → VAD:  _AudioChunk (samples)
//   VAD → UI:  _SpeechState (isSpeech)
//   VAD → UI:  _SegmentDetected (index, start, end, samples) — shown as "Decoding..."
//   VAD → ASR: _AsrSegmentRequest (index, samples, sampleRate, start, end)
//   ASR → UI:  _SegmentFound (index, start, end, samples, text, elapsedSeconds)
//              → emits TextUpdate to update segment text + RTF
import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './model_config.dart' as cfg;

/// A VAD segment with ASR transcription.
class VadAsrSegment {
  final double start; // seconds
  final double end; // seconds
  final Float32List samples;
  String text; // mutable — updated when ASR finishes
  double elapsedSeconds; // ASR decoding time (mutable)

  VadAsrSegment({
    required this.start,
    required this.end,
    required this.samples,
    this.text = '',
    this.elapsedSeconds = 0,
  });
}

enum VadAsrState { uninitialized, initializing, initialized }

class VadAsrManager {
  VadAsrState _state = VadAsrState.uninitialized;
  VadAsrState get state => _state;

  final _logController = StreamController<String>.broadcast();
  final _segmentController = StreamController<VadAsrSegment>.broadcast();
  final _textUpdateController = StreamController<TextUpdate>.broadcast();
  final _speechController = StreamController<bool>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<VadAsrSegment> get segmentStream => _segmentController.stream;
  Stream<TextUpdate> get textUpdateStream => _textUpdateController.stream;
  Stream<bool> get speechStream => _speechController.stream;

  Isolate? _vadIsolate;
  Isolate? _asrIsolate;
  SendPort? _vadSendPort;
  ReceivePort? _vadReceivePort;
  ReceivePort? _asrReceivePort;

  Future<void> init({
    required String modelDir,
    required String vadModelDir,
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    if (_state != VadAsrState.uninitialized) return;
    _state = VadAsrState.initializing;
    _logController.add('Initializing VAD + ASR...');

    try {
      _vadReceivePort = ReceivePort();
      _asrReceivePort = ReceivePort();

      // Spawn ASR isolate first.
      _asrIsolate = await Isolate.spawn(
        _asrIsolateEntry,
        _AsrInitRequest(
          readyPort: _asrReceivePort!.sendPort,
          resultPort: _vadReceivePort!.sendPort,
          modelDir: modelDir,
        ),
      );

      // Wait for ASR isolate to be ready.
      final asrSegmentPort = await _asrReceivePort!.first.timeout(
        const Duration(seconds: 30),
        onTimeout: () => throw TimeoutException('ASR isolate init timed out'),
      ) as SendPort;

      // Set up listener first, then spawn VAD.
      final vadReady = Completer<void>();
      bool initDone = false;

      _vadReceivePort!.listen((message) {
        if (!initDone) {
          initDone = true;
          if (message is SendPort) {
            _vadSendPort = message;
            vadReady.complete();
          } else {
            vadReady.completeError(Exception('Unexpected first VAD message: $message'));
          }
        } else {
          _onWorkerMessage(message);
        }
      });

      // Spawn VAD isolate with the correct VAD model directory and user params.
      _vadIsolate = await Isolate.spawn(
        _vadIsolateEntry,
        _VadInitRequest(
          asrSegmentPort: asrSegmentPort,
          uiSendPort: _vadReceivePort!.sendPort,
          modelDir: vadModelDir,
          threshold: threshold,
          minSilenceDuration: minSilenceDuration,
          minSpeechDuration: minSpeechDuration,
          maxSpeechDuration: maxSpeechDuration,
        ),
      );

      await vadReady.future.timeout(
        const Duration(seconds: 30),
        onTimeout: () => throw TimeoutException('VAD isolate init timed out'),
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
    if (message is _SegmentDetected) {
      // VAD detected a speech segment — show immediately with "Decoding..."
      final seg = VadAsrSegment(
        start: message.start,
        end: message.end,
        samples: message.samples,
        text: 'Decoding...',
      );
      _segmentController.add(seg);
    } else if (message is _SegmentFound) {
      final audioDuration = message.end - message.start;
      final rtf = audioDuration > 0
          ? message.elapsedSeconds / audioDuration
          : 0.0;
      print('ASR: segment #${message.index} → "${message.text}" '
          '(audio: ${audioDuration.toStringAsFixed(2)}s, '
          'elapsed: ${message.elapsedSeconds.toStringAsFixed(3)}s, '
          'RTF: ${rtf.toStringAsFixed(3)})');
      // ASR finished — update the segment text.
      _textUpdateController.add(TextUpdate(message.index, message.text, message.elapsedSeconds));
    } else if (message is _SpeechState) {
      _speechController.add(message.isSpeech);
    } else if (message is _LogMessage) {
      _logController.add(message.text);
    }
  }

  /// Send a chunk of audio samples to the VAD + ASR pipeline.
  void acceptWaveform(Float32List samples) {
    if (_state != VadAsrState.initialized) return;
    _vadSendPort!.send(_AudioChunk(samples));
  }

  /// Reset VAD state for a new recording session. ASR isolate stays alive.
  void reset() {
    _vadSendPort?.send(_Reset());
  }

  void cancel() {
    _vadSendPort?.send(_Cancel());
  }

  void dispose() {
    _vadIsolate?.kill(priority: Isolate.immediate);
    _asrIsolate?.kill(priority: Isolate.immediate);
    _vadReceivePort?.close();
    _asrReceivePort?.close();
    _logController.close();
    _segmentController.close();
    _textUpdateController.close();
    _speechController.close();
  }
}

// ============================================================
// Messages
// ============================================================

class _AudioChunk {
  final Float32List samples;
  const _AudioChunk(this.samples);
}

class _Cancel {}

class _Reset {}

class _AsrSegmentRequest {
  final int index;
  final Float32List samples;
  final int sampleRate;
  final double start;
  final double end;
  const _AsrSegmentRequest({
    required this.index,
    required this.samples,
    required this.sampleRate,
    required this.start,
    required this.end,
  });
}

class _SegmentFound {
  final int index;
  final double start;
  final double end;
  final Float32List samples;
  final String text;
  final double elapsedSeconds; // ASR decoding time
  const _SegmentFound({
    required this.index,
    required this.start,
    required this.end,
    required this.samples,
    required this.text,
    required this.elapsedSeconds,
  });
}

class _SpeechState {
  final bool isSpeech;
  const _SpeechState(this.isSpeech);
}

class _LogMessage {
  final String text;
  const _LogMessage(this.text);
}

/// Sent from VAD isolate to UI when a speech segment is detected (before ASR).
class _SegmentDetected {
  final int index;
  final double start;
  final double end;
  final Float32List samples;
  const _SegmentDetected(this.index, this.start, this.end, this.samples);
}

/// ASR text update for a segment.
class TextUpdate {
  final int index;
  final String text;
  final double elapsedSeconds; // ASR decoding time
  const TextUpdate(this.index, this.text, this.elapsedSeconds);
}

// ============================================================
// VAD Isolate
// ============================================================

class _VadInitRequest {
  final SendPort asrSegmentPort;
  final SendPort uiSendPort;
  final String modelDir;
  final double threshold;
  final double minSilenceDuration;
  final double minSpeechDuration;
  final double maxSpeechDuration;
  const _VadInitRequest({
    required this.asrSegmentPort,
    required this.uiSendPort,
    required this.modelDir,
    required this.threshold,
    required this.minSilenceDuration,
    required this.minSpeechDuration,
    required this.maxSpeechDuration,
  });
}

void _vadIsolateEntry(_VadInitRequest initReq) async {
  await sherpa_onnx.initBindingsAsync();

  final baseVadConfig = cfg.defaultVadConfig;
  final vadConfig = sherpa_onnx.VadModelConfig(
    sileroVad: sherpa_onnx.SileroVadModelConfig(
      model: '${initReq.modelDir}/${baseVadConfig.sileroVad.model}',
      threshold: initReq.threshold,
      minSilenceDuration: initReq.minSilenceDuration,
      minSpeechDuration: initReq.minSpeechDuration,
      windowSize: baseVadConfig.sileroVad.windowSize,
      maxSpeechDuration: initReq.maxSpeechDuration,
    ),
    numThreads: baseVadConfig.numThreads,
    debug: baseVadConfig.debug,
  );
  final vad = sherpa_onnx.VoiceActivityDetector(
    config: vadConfig,
    bufferSizeInSeconds: 300,
  );

  final receivePort = ReceivePort();

  // Send our SendPort back to UI.
  initReq.uiSendPort.send(receivePort.sendPort);

  int segmentIndex = 0;

  receivePort.listen((message) {
    if (message is _Reset) {
      // Reset VAD state for a new recording session.
      vad.reset();
      segmentIndex = 0;
    } else if (message is _AudioChunk) {
      vad.acceptWaveform(message.samples);

      // Check speech state.
      final isSpeech = vad.isDetected();
      initReq.uiSendPort.send(_SpeechState(isSpeech));

      // Process completed segments — notify UI immediately, then send to ASR.
      while (!vad.isEmpty()) {
        final segSamples = vad.front().samples;
        const sampleRate = 16000; // Always 16kHz from microphone
        final startSec = vad.front().start / sampleRate;
        final endSec = startSec + segSamples.length / sampleRate;
        final idx = segmentIndex++;

        print('VAD: segment #$idx detected: ${startSec.toStringAsFixed(2)}-${endSec.toStringAsFixed(2)}s (${segSamples.length} samples)');

        // Tell UI a segment was detected (shows "Decoding...").
        initReq.uiSendPort.send(_SegmentDetected(idx, startSec, endSec, segSamples));

        // Send to ASR isolate for transcription.
        initReq.asrSegmentPort.send(_AsrSegmentRequest(
          index: idx,
          samples: segSamples,
          sampleRate: sampleRate,
          start: startSec,
          end: endSec,
        ));

        vad.pop();
      }
    }
  });
}

// ============================================================
// ASR Isolate
// ============================================================

class _AsrInitRequest {
  final SendPort readyPort;
  final SendPort resultPort;
  final String modelDir;
  const _AsrInitRequest({
    required this.readyPort,
    required this.resultPort,
    required this.modelDir,
  });
}

void _asrIsolateEntry(_AsrInitRequest initReq) async {
  await sherpa_onnx.initBindingsAsync();

  final asrConfig = cfg.buildAsrConfig(modelDir: initReq.modelDir);
  final recognizer = sherpa_onnx.OfflineRecognizer(asrConfig);

  final receivePort = ReceivePort();

  // Send our SendPort back to UI so VAD can send us segments.
  initReq.readyPort.send(receivePort.sendPort);

  receivePort.listen((message) {
    if (message is _AsrSegmentRequest) {
      final sw = Stopwatch()..start();
      final stream = recognizer.createStream();
      stream.acceptWaveform(
          samples: message.samples, sampleRate: message.sampleRate);
      recognizer.decode(stream);
      final result = recognizer.getResult(stream);
      stream.free();
      sw.stop();

      // Send result back to UI.
      initReq.resultPort.send(_SegmentFound(
        index: message.index,
        start: message.start,
        end: message.end,
        samples: message.samples,
        text: result.text,
        elapsedSeconds: sw.elapsedMilliseconds / 1000.0,
      ));
    }
  });
}
