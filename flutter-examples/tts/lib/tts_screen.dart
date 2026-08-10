// Copyright (c)  2026  Xiaomi Corporation
import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';

import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import 'package:audioplayers/audioplayers.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

import './generated_audio.dart';
import './tts_manager.dart';
import './tts_controls.dart';
import './audio_list.dart';
import './web_audio.dart' if (dart.library.io) './web_audio_stub.dart'
    as web_audio;
import './save_file.dart' if (dart.library.js_interop) './save_file_stub.dart'
    as save_file;
import './play_bytes.dart' if (dart.library.js_interop) './play_bytes_stub.dart'
    as play_bytes;
import './play_ref.dart' if (dart.library.js_interop) './play_ref_stub.dart'
    as play_ref;
import './wav_encoder.dart';
import './model_config.dart' show selectedModelIndex;
import 'package:file_picker/file_picker.dart';

class TtsScreen extends StatefulWidget {
  const TtsScreen({super.key});

  @override
  State<TtsScreen> createState() => _TtsScreenState();
}

class _TtsScreenState extends State<TtsScreen> {
  final _textController = TextEditingController();
  final _sidController = TextEditingController(text: '0');
  final _logController = TextEditingController();
  final _numStepsController = TextEditingController(text: '5');

  late final TtsManager _manager;
  AudioPlayer? _player;

  final List<GeneratedAudioItem> _audioItems = [];
  int _maxSpeakerID = 0;
  double _speed = 1.0;
  bool _isGenerating = false;

  // Reference audio for Pocket TTS voice cloning.
  static const bool _isPocketTts = selectedModelIndex == 9;
  Float32List? _referenceAudio;
  int _referenceSampleRate = 0;
  String? _referenceAudioName;
  bool _isRefPlaying = false;
  // On native: incremented on each Generate/Stop to ignore stale chunks.
  // On web: unused (worker is terminated on Stop, no stale chunks).
  int _generationId = 0;

  double _generationProgress = 0.0;

  // Streaming playback (native).
  final List<Float32List> _chunkBuffer = [];
  int _chunkSampleRate = 0;
  static const int _chunkThresholdSamples = 16000; // ~1s at 16kHz

  // Queue of encoded WAV segments waiting to be played.
  final Queue<Uint8List> _playQueue = Queue();
  bool _isPlayingSegment = false;

  @override
  void initState() {
    super.initState();
    _manager = TtsManager();

    if (!kIsWeb) {
      _player = AudioPlayer();
      // Listen for playback completion to play next queued segment.
      _player!.onPlayerComplete.listen((_) {
        _isPlayingSegment = false;
        _playNextSegment();
        if (mounted && _isRefPlaying) {
          setState(() => _isRefPlaying = false);
        }
      });
    }

    _manager.logStream.listen((msg) {
      if (mounted) {
        setState(() => _logController.text = msg);
      }
    });

    _manager.initStream.listen((_) {
      if (mounted) {
        setState(() {
          _maxSpeakerID = _manager.numSpeakers;
          if (_maxSpeakerID > 0) _maxSpeakerID--;
        });
      }
    });

    // Stream audio chunks for real-time playback.
    _manager.chunkStream.listen((chunk) {
      if (!mounted) return;
      // On native: ignore chunks from a previous generation.
      if (!kIsWeb && chunk.generationId != _generationId) return;

      setState(() {
        _generationProgress = chunk.progress;
        _logController.text =
            'Generating... ${(chunk.progress * 100).toStringAsFixed(0)}%';
      });

      if (kIsWeb) {
        web_audio.playAudioChunk(chunk.samples, chunk.sampleRate);
      } else {
        _chunkBuffer.add(chunk.samples);
        _chunkSampleRate = chunk.sampleRate;

        int totalSamples = 0;
        for (final c in _chunkBuffer) {
          totalSamples += c.length;
        }

        if (totalSamples >= _chunkThresholdSamples) {
          _flushChunkBuffer();
        }
      }
    });

    // When generation completes, flush remaining and add to list.
    _manager.audioStream.listen((item) {
      if (!mounted) return;
      // On native: ignore results from a previous generation.
      if (!kIsWeb && item.generationId != _generationId) return;

      _generationProgress = 0.0;
      if (!kIsWeb && _chunkBuffer.isNotEmpty) {
        _flushChunkBuffer();
      }
      _chunkBuffer.clear();

      final rtf = item.elapsed / item.duration;
      final status = 'Duration: ${item.duration.toStringAsFixed(2)}s\n'
          'Elapsed: ${item.elapsed.toStringAsFixed(2)}s\n'
          'RTF: ${item.elapsed.toStringAsFixed(2)} / ${item.duration.toStringAsFixed(2)} = ${rtf.toStringAsFixed(3)}';

      setState(() {
        _isGenerating = false;
        _audioItems.insert(0, item);
        _logController.text = status;
      });
    });
  }

  /// Encode buffered chunks as WAV and enqueue for sequential playback.
  void _flushChunkBuffer() {
    if (_chunkBuffer.isEmpty) return;

    int total = 0;
    for (final c in _chunkBuffer) {
      total += c.length;
    }
    final merged = Float32List(total);
    int offset = 0;
    for (final c in _chunkBuffer) {
      merged.setRange(offset, offset + c.length, c);
      offset += c.length;
    }
    _chunkBuffer.clear();

    final wavBytes = encodeWav(merged, _chunkSampleRate);
    _playQueue.add(wavBytes);
    _playNextSegment();
  }

  /// Play the next segment from the queue if not already playing.
  Future<void> _playNextSegment() async {
    if (_isPlayingSegment || _playQueue.isEmpty || _player == null) return;
    _isPlayingSegment = true;
    final wavBytes = _playQueue.removeFirst();
    try {
      await play_bytes.playWavBytes(_player!, wavBytes);
    } catch (_) {
      _isPlayingSegment = false;
    }
  }

  Future<void> _initIfNeeded() async {
    if (_manager.state != TtsState.uninitialized) return;
    try {
      await _manager.init();
    } catch (_) {}
  }

  Future<void> _onPickReferenceAudio() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['wav'],
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;
    final bytes = file.bytes;
    if (bytes == null) {
      setState(() => _logController.text = 'Error: Could not read file');
      return;
    }

    final wav = decodeWav(bytes);
    if (wav == null) {
      setState(() => _logController.text =
          'Error: Unsupported WAV format (need 16-bit PCM or 32-bit float)');
      return;
    }
    setState(() {
      _referenceAudio = wav.samples;
      _referenceSampleRate = wav.sampleRate;
      _referenceAudioName = file.name;
    });
  }

  Future<void> _playReferenceAudio() async {
    if (_referenceAudio == null || _referenceAudio!.isEmpty) return;
    final wavBytes = encodeWav(_referenceAudio!, _referenceSampleRate);
    setState(() => _isRefPlaying = true);
    if (kIsWeb) {
      web_audio.playWavBytes(wavBytes);
      // Web Audio API doesn't have a completion callback easily;
      // reset after a rough duration estimate.
      final duration = _referenceAudio!.length / _referenceSampleRate;
      Future.delayed(Duration(milliseconds: (duration * 1000).ceil() + 200), () {
        if (mounted) setState(() => _isRefPlaying = false);
      });
    } else {
      if (_player != null) {
        await play_ref.playRefWavBytes(_player!, wavBytes);
      }
    }
  }

  Future<void> _stopReferenceAudio() async {
    if (kIsWeb) {
      web_audio.stopPlayback();
    } else {
      await _player?.stop();
    }
    setState(() => _isRefPlaying = false);
  }

  Future<void> _playAudio(GeneratedAudioItem item) async {
    if (kIsWeb) {
      web_audio.playWavBytes(item.wavBytes!);
    } else {
      await _player?.stop();
      await _player?.play(DeviceFileSource(item.filePath!));
    }
  }

  Future<void> _onGenerate() async {
    await _initIfNeeded();

    if (!kIsWeb) {
      await _player?.stop();
      _isPlayingSegment = false;
      _chunkBuffer.clear();
      _playQueue.clear();
    }
    if (_isRefPlaying) {
      setState(() => _isRefPlaying = false);
    }

    final text = _textController.text.trim();
    if (text.isEmpty) {
      setState(() => _logController.text = 'Please enter text to synthesize');
      return;
    }

    if (_isPocketTts && _referenceAudio == null) {
      setState(() => _logController.text = 'Please pick a reference WAV file first');
      return;
    }

    final sid = int.tryParse(_sidController.text.trim()) ?? 0;

    if (!kIsWeb) _generationId++;
    if (kIsWeb) {
      web_audio.resetChunkPlayback();
    }
    final numSteps = int.tryParse(_numStepsController.text.trim()) ?? 5;
    final id = _manager.generate(
      text: text, sid: sid, speed: _speed, generationId: _generationId,
      referenceAudio: _isPocketTts ? _referenceAudio : null,
      referenceSampleRate: _isPocketTts ? _referenceSampleRate : 0,
      numSteps: _isPocketTts ? numSteps : 5,
    );
    if (id < 0) {
      // Generate failed (not initialized).
      return;
    }
    setState(() => _isGenerating = true);
  }

  Future<void> _onSaveAs(GeneratedAudioItem item, int index) async {
    if (kIsWeb) {
      final controller = TextEditingController(text: '$index-${item.label}.wav');
      final filename = await showDialog<String>(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('Save as'),
          content: TextField(
            controller: controller,
            decoration: const InputDecoration(
              labelText: 'Filename',
              border: OutlineInputBorder(),
            ),
            autofocus: true,
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () =>
                  Navigator.pop(context, controller.text.trim()),
              child: const Text('Save'),
            ),
          ],
        ),
      );
      if (filename != null && filename.isNotEmpty) {
        web_audio.downloadWavBytes(item.wavBytes!, filename);
      }
    } else {
      try {
        final savedPath =
            await save_file.saveFileAs(item.filePath!, '${item.label}.wav');
        if (savedPath != null && mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Saved to $savedPath')),
          );
        }
      } catch (e) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error: $e')),
          );
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Text to Speech')),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            TtsControls(
              maxSpeakerID: _maxSpeakerID,
              speed: _speed,
              onSpeedChanged: (v) => setState(() => _speed = v),
              textController: _textController,
              sidController: _sidController,
              onGenerate: _onGenerate,
              onClear: () {
                _textController.clear();
                _logController.clear();
              },
              showReferenceAudio: _isPocketTts,
              referenceAudioLabel: _referenceAudioName,
              onPickReferenceAudio: _onPickReferenceAudio,
              onPlayReferenceAudio: _playReferenceAudio,
              onStopReferenceAudio: _stopReferenceAudio,
              isRefPlaying: _isRefPlaying,
              numStepsController: _numStepsController,
              onStop: () {
                // Stop generation.
                // On native: increment generationId to invalidate stale chunks.
                // On web: worker is terminated, no stale chunks possible.
                _manager.cancel();
                if (!kIsWeb) _generationId++;
                // Stop playback and clear queues.
                if (kIsWeb) {
                  web_audio.stopPlayback();
                  web_audio.resetChunkPlayback();
                } else {
                  _player?.stop();
                  _isPlayingSegment = false;
                  _chunkBuffer.clear();
                  _playQueue.clear();
                }
                setState(() {
                  _isGenerating = false;
                  _generationProgress = 0.0;
                });
              },
              isGenerating: _isGenerating ||
                  _manager.state == TtsState.initializing,
            ),
            const SizedBox(height: 4),
            TextField(
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Status',
                isDense: true,
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              ),
              maxLines: 3,
              controller: _logController,
              readOnly: true,
            ),
            if (_audioItems.isNotEmpty) ...[
              const SizedBox(height: 4),
              Expanded(
                child: AudioList(
                  items: _audioItems,
                  player: _player,
                  onSaveAs: _onSaveAs,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _manager.dispose();
    _player?.dispose();
    _textController.dispose();
    _sidController.dispose();
    _logController.dispose();
    _numStepsController.dispose();
    super.dispose();
  }
}
