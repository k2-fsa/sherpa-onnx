// Copyright (c)  2026  Xiaomi Corporation
//
// TTS manager — handles TTS lifecycle for both native and web.
//
// On native: communicates with a background isolate via SendPort/ReceivePort.
// On web: delegates to TtsWorker which communicates with a Web Worker.
//
// See worker_web.dart and tts-worker.js for the web message protocol.
//
// ── Native Isolate Message Protocol ────────────────────────────────────────
//
// The main isolate and background isolate communicate over a pair of
// SendPort/ReceivePort. The very first message from the background isolate
// is its SendPort (bidirectional setup). After that, messages flow as typed
// Dart objects.
//
// Main → Background:
//
//   OfflineTtsConfig   — initial config; triggers TTS creation.
//                        The background isolate calls sherpa_onnx.OfflineTts(config)
//                        and replies with _Ready or _WorkerError.
//
//   _GenerateRequest   — synthesize speech.
//     .text              String       — text to synthesize
//     .sid               int          — speaker ID
//     .speed             double       — speech rate
//     .generationId      int          — id for matching chunks/result
//     .referenceAudio    Float32List? — optional PCM samples for voice cloning
//     .referenceSampleRate int        — sample rate of reference audio
//     .numSteps          int          — diffusion steps (Pocket TTS)
//
//   _DisposeRequest    — free the OfflineTts and close the isolate.
//
// Background → Main:
//
//   SendPort         — first message; the main isolate uses this to send back.
//   SendPort         — second+ messages; per-generation cancel port.
//                      Sent each time _handleGenerate starts.
//                      Main isolate sends `true` on this port to cancel.
//
//   _Ready           — TTS created successfully.
//     .numSpeakers      int
//
//   _AudioChunk      — streaming audio chunk (sent during generation).
//     .samples          Float32List  — PCM samples
//     .progress         double       — 0.0–1.0
//     .sampleRate       int
//     .generationId     int
//
//   _GenerateDone    — generation complete.
//     .samples          Float32List  — full PCM audio
//     .sampleRate       int
//     .duration         double       — audio length in seconds
//     .elapsed          double       — wall-clock time in seconds
//     .generationId     int
//
//   _WorkerLog       — debug/info message.
//     .message          String
//
//   _WorkerError     — error message.
//     .message          String
//
// ───────────────────────────────────────────────────────────────────────────

import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kDebugMode, kIsWeb;
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './generated_audio.dart';
import './model.dart' if (dart.library.js_interop) './model_web.dart' as m;
import './utils.dart' if (dart.library.js_interop) './utils_web.dart' as u;
import './web_audio.dart' if (dart.library.io) './web_audio_stub.dart'
    as web_audio;
import './worker_web.dart' if (dart.library.io) './worker_stub.dart'
    as worker_lib;

/// State of the TTS engine.
enum TtsState { uninitialized, initializing, initialized }

// ── Messages from main isolate → background isolate (native only) ────────

sealed class _ToWorker {}

class _GenerateRequest extends _ToWorker {
  final String text;
  final int sid;
  final double speed;
  final int generationId;
  final Float32List? referenceAudio;
  final int referenceSampleRate;
  final int numSteps;
  _GenerateRequest(this.text, this.sid, this.speed, this.generationId,
      {this.referenceAudio, this.referenceSampleRate = 0, this.numSteps = 5});
}

class _DisposeRequest extends _ToWorker {}

// ── Messages from background isolate → main isolate (native only) ────────

sealed class _FromWorker {}

class _Ready extends _FromWorker {
  final int numSpeakers;
  _Ready(this.numSpeakers);
}

class _GenerateDone extends _FromWorker {
  final Float32List samples;
  final int sampleRate;
  final double duration;
  final double elapsed;
  final int generationId;
  _GenerateDone(this.samples, this.sampleRate, this.duration, this.elapsed,
      this.generationId);
}

class _AudioChunk extends _FromWorker {
  final Float32List samples;
  final double progress;
  final int sampleRate;
  final int generationId;
  _AudioChunk(this.samples, this.progress, this.sampleRate, this.generationId);
}

class _WorkerError extends _FromWorker {
  final String message;
  _WorkerError(this.message);
}

class _WorkerLog extends _FromWorker {
  final String message;
  _WorkerLog(this.message);
}

// ── Pending generate tracking ────────────────────────────────────────────

class _PendingGenerate {
  final String text;
  final int sid;
  final double speed;
  final int generationId;
  _PendingGenerate({
    required this.text,
    required this.sid,
    required this.speed,
    this.generationId = 0,
  });
}

// ── TtsManager ───────────────────────────────────────────────────────────

/// Manages TTS lifecycle with isolate-based execution on native
/// and Web Worker-based execution on web.
class TtsManager {
  final _logController = StreamController<String>.broadcast();
  final _audioController = StreamController<GeneratedAudioItem>.broadcast();
  final _initController = StreamController<void>.broadcast();
  final _chunkController = StreamController<AudioChunk>.broadcast();

  Stream<String> get logStream => _logController.stream;
  Stream<GeneratedAudioItem> get audioStream => _audioController.stream;
  Stream<void> get initStream => _initController.stream;
  Stream<AudioChunk> get chunkStream => _chunkController.stream;

  // Native: isolate-based.
  Isolate? _isolate;
  SendPort? _sendPort;
  final Map<int, _PendingGenerate> _pending = {};

  // Web: Web Worker-based.
  worker_lib.TtsWorker? _worker;

  TtsState _state = TtsState.uninitialized;
  int _numSpeakers = 0;
  int _nextId = 0;

  int get numSpeakers => _numSpeakers;
  TtsState get state => _state;
  bool get isInitialized => _state == TtsState.initialized;

  /// Initialize the TTS engine.
  Future<void> init() async {
    if (_state != TtsState.uninitialized) return;
    _state = TtsState.initializing;

    if (kIsWeb) {
      await _initWeb();
    } else {
      await _initNative();
    }
  }

  /// Generate audio from text.
  int generate({
    required String text,
    int sid = 0,
    double speed = 1.0,
    int generationId = 0,
    Float32List? referenceAudio,
    int referenceSampleRate = 0,
    int numSteps = 5,
  }) {
    if (_state != TtsState.initialized) {
      _logController.add('Error: TTS not initialized');
      return -1;
    }

    if (kDebugMode) {
      print('[tts_manager] generate: text="$text", sid=$sid, speed=$speed');
    }

    final id = _nextId++;

    if (kIsWeb) {
      _worker?.generate(
        text: text, sid: sid, speed: speed, generationId: generationId,
        referenceAudio: referenceAudio, referenceSampleRate: referenceSampleRate,
        numSteps: numSteps,
      );
    } else {
      _pending[id] = _PendingGenerate(
        text: text, sid: sid, speed: speed, generationId: generationId,
      );
      _sendPort!.send(_GenerateRequest(text, sid, speed, generationId,
        referenceAudio: referenceAudio, referenceSampleRate: referenceSampleRate,
        numSteps: numSteps,
      ));
    }

    return id;
  }

  /// Cancel the current generation.
  /// On web: terminates the worker (TTS is recreated on next Generate).
  /// On native: sends cancel signal to the isolate (TTS stays alive).
  void cancel() {
    if (kIsWeb) {
      // The WASM call blocks the worker, so cancel messages are queued
      // and never processed. Terminate the worker instead.
      _worker?.dispose();
      _worker = null;
      _state = TtsState.uninitialized;
    } else {
      // Send cancel signal to the isolate.
      // The callback checks this and returns 0 to stop generation.
      _cancelPort?.send(true);
    }
    _pending.clear();
  }

  /// Port for sending cancel signals to the background isolate.
  SendPort? _cancelPort;

  /// Dispose the TTS engine and release resources.
  void dispose() {
    _state = TtsState.uninitialized;
    if (kIsWeb) {
      _worker?.dispose();
      _worker = null;
    } else {
      _sendPort?.send(_DisposeRequest());
      _isolate?.kill(priority: Isolate.immediate);
      _isolate = null;
      _sendPort = null;
    }
    _logController.close();
    _audioController.close();
    _initController.close();
    _chunkController.close();
  }

  // ── Web (Web Worker) ──────────────────────────────────────────────────

  Future<void> _initWeb() async {
    try {
      final readyCompleter = Completer<void>();

      _worker = worker_lib.TtsWorker(
        onReady: (numSpeakers) {
          _numSpeakers = numSpeakers;
          _state = TtsState.initialized;
          _logController.add('TTS ready (speakers: $_numSpeakers)');
          _initController.add(null);
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        },
        onChunk: (chunk) {
          _chunkController.add(chunk);
        },
        onDone: (item) {
          _audioController.add(item);
        },
        onError: (msg) {
          _state = TtsState.uninitialized;
          _logController.add('Error: $msg');
          if (!readyCompleter.isCompleted) readyCompleter.completeError(msg);
        },
      );

      await _worker!.init();
      await readyCompleter.future;
    } catch (e) {
      _state = TtsState.uninitialized;
      _logController.add('Error: $e');
    }
  }

  // ── Native (isolate-based) ──────────────────────────────────────────────

  Future<void> _initNative() async {
    try {
      _logController.add('Preparing model...');
      final config = await m.prepareModelConfig();

      if (kDebugMode) {
        print('[tts_manager] config: $config');
      }

      // IMPORTANT: sherpa-onnx must be initialized in every isolate that calls
      // its Dart API. The main isolate needs initBindings() for writeWave(),
      // AudioPlayer, and other operations that happen after generation.
      sherpa_onnx.initBindings();

      _logController.add('Starting TTS isolate...');
      final receivePort = ReceivePort();
      _isolate = await Isolate.spawn(_workerEntry, receivePort.sendPort);

      final readyCompleter = Completer<void>();

      bool gotSendPort = false;
      receivePort.listen((message) {
        if (message is SendPort && !gotSendPort) {
          // First SendPort: main communication port.
          gotSendPort = true;
          _sendPort = message;
          message.send(config);
        } else if (message is SendPort && gotSendPort) {
          // Per-generation cancel port (sent each time _handleGenerate runs).
          _cancelPort = message;
        } else if (message is _Ready) {
          _numSpeakers = message.numSpeakers;
          _state = TtsState.initialized;
          _logController.add('TTS ready (speakers: $_numSpeakers)');
          _initController.add(null);
          if (!readyCompleter.isCompleted) readyCompleter.complete();
        } else if (message is _AudioChunk) {
          _chunkController.add(AudioChunk(
            samples: Float32List.fromList(message.samples),
            progress: message.progress,
            sampleRate: message.sampleRate,
            generationId: message.generationId,
          ));
        } else if (message is _GenerateDone) {
          _handleGenerateDone(message);
        } else if (message is _WorkerLog) {
          _logController.add(message.message);
        } else if (message is _WorkerError) {
          _state = TtsState.uninitialized;
          _logController.add('Error: ${message.message}');
          if (!readyCompleter.isCompleted) {
            readyCompleter.completeError(message.message);
          }
        }
      });

      return readyCompleter.future;
    } catch (e) {
      _state = TtsState.uninitialized;
      _logController.add('Error: $e');
      rethrow;
    }
  }

  static void _workerEntry(SendPort mainSendPort) {
    final receivePort = ReceivePort();
    mainSendPort.send(receivePort.sendPort);

    sherpa_onnx.OfflineTts? tts;

    receivePort.listen((message) {
      if (message is sherpa_onnx.OfflineTtsConfig) {
        try {
          // IMPORTANT: sherpa-onnx must be initialized in every isolate.
          sherpa_onnx.initBindings();
          tts = sherpa_onnx.OfflineTts(message);
          mainSendPort.send(_Ready(tts!.numSpeakers));
        } catch (e) {
          mainSendPort.send(_WorkerError('$e'));
        }
      } else if (message is _GenerateRequest && tts != null) {
        _handleGenerate(mainSendPort, tts!, message);
      } else if (message is _DisposeRequest) {
        tts?.free();
        tts = null;
        receivePort.close();
      }
    });
  }

  static void _handleGenerate(SendPort mainSendPort, sherpa_onnx.OfflineTts tts,
      _GenerateRequest req) {
    // Create a fresh cancel port for each generation.
    final cancelPort = ReceivePort();
    mainSendPort.send(cancelPort.sendPort);

    try {
      final stopwatch = Stopwatch()..start();
      bool cancelled = false;

      cancelPort.listen((_) {
        cancelled = true;
      });

      final genConfig = sherpa_onnx.OfflineTtsGenerationConfig(
        sid: req.sid,
        speed: req.speed,
        silenceScale: 0.2,
        referenceAudio: req.referenceAudio,
        referenceSampleRate: req.referenceSampleRate,
        numSteps: req.numSteps,
      );

      final sampleRate = tts.sampleRate;
      final genId = req.generationId;
      final audio = tts.generateWithConfig(
        text: req.text,
        config: genConfig,
        onProgress: (samples, progress) {
          if (cancelled) return 0; // stop generation
          mainSendPort.send(_AudioChunk(
            Float32List.fromList(samples),
            progress,
            sampleRate,
            genId,
          ));
          return 1; // continue generation
        },
      );

      stopwatch.stop();
      final elapsed = stopwatch.elapsedMilliseconds / 1000.0;
      final duration = audio.samples.length / audio.sampleRate;

      mainSendPort.send(_GenerateDone(
        Float32List.fromList(audio.samples),
        audio.sampleRate,
        duration,
        elapsed,
        genId,
      ));
    } catch (e) {
      mainSendPort.send(_WorkerError('$e'));
    } finally {
      cancelPort.close();
    }
  }

  void _handleGenerateDone(_GenerateDone msg) async {
    if (_pending.isEmpty) return;
    final entry = _pending.entries.first;
    _pending.remove(entry.key);
    final text = entry.value.text;

    final label = GeneratedAudioItem.makeLabel(text);
    final suffix =
        '-sid-${entry.value.sid}-speed-${entry.value.speed.toStringAsPrecision(2)}';
    final filename = await u.generateWaveFilename(suffix);
    final ok = sherpa_onnx.writeWave(
      filename: filename,
      samples: msg.samples,
      sampleRate: msg.sampleRate,
    );

    if (ok) {
      _audioController.add(GeneratedAudioItem(
        label: label,
        filePath: filename,
        duration: msg.duration,
        elapsed: msg.elapsed,
        sampleRate: msg.sampleRate,
        generationId: msg.generationId,
      ));
    }
  }
}
