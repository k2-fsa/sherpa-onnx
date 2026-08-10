// Copyright (c)  2026  Xiaomi Corporation
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:video_player/video_player.dart';

import './audio_decoder_native.dart'
    if (dart.library.js_interop) './audio_decoder_web.dart';
import './cleanup_native.dart'
    if (dart.library.js_interop) './cleanup_stub.dart';
import './model_config.dart' as cfg;
import './vad_controls.dart';
import './vad_manager.dart';
import './wav_encoder.dart';
import './save_file.dart' if (dart.library.js_interop) './save_file_stub.dart'
    as save_file;
import './web_audio.dart' if (dart.library.io) './web_audio_stub.dart'
    as web_audio;
import './web_blob_url.dart' if (dart.library.io) './web_blob_url_stub.dart';

class VadScreen extends StatefulWidget {
  const VadScreen({super.key});

  @override
  State<VadScreen> createState() => _VadScreenState();
}

class _VadScreenState extends State<VadScreen> {
  final _logController = TextEditingController();
  final _thresholdController = TextEditingController(text: '0.1');
  final _minSilenceController = TextEditingController(text: '0.5');
  final _minSpeechController = TextEditingController(text: '0.25');
  final _maxSpeechController = TextEditingController(text: '12.0');

  late final VadManager _manager;
  VideoPlayerController? _playerController;

  List<VadSegment> _segments = [];
  bool _isProcessing = false;
  double _progress = 0.0;
  double _elapsed = 0.0;
  double _audioDuration = 0.0;
  String? _fileName;
  Uint8List? _fileBytes; // Raw file bytes for VAD decoding.
  int _playingIndex = -1;
  bool _playAllFromSegment = false;

  // Playback position tracking.
  Duration _currentPosition = Duration.zero;
  Duration _totalDuration = Duration.zero;
  bool _isPlaying = false;

  Float32List? _loadedSamples;
  int _loadedSampleRate = 16000;

  @override
  void initState() {
    super.initState();
    _manager = VadManager();

    _manager.logStream.listen((msg) {
      if (mounted) setState(() => _logController.text = msg);
    });

    _manager.progressStream.listen((progress) {
      if (mounted) setState(() => _progress = progress);
    });

    _manager.resultStream.listen((result) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _progress = 0.0;
        _segments = result.segments;
        _elapsed = result.elapsed;
        _audioDuration = result.audioDuration;
        final rtf = _audioDuration > 0 ? _elapsed / _audioDuration : 0.0;
        _logController.text =
            'Duration: ${_audioDuration.toStringAsFixed(2)}s | '
            'Elapsed: ${_elapsed.toStringAsFixed(2)}s | '
            'RTF: ${rtf.toStringAsFixed(3)} (${_elapsed.toStringAsFixed(2)} / ${_audioDuration.toStringAsFixed(2)})\n'
            'Segments: ${_segments.length}';
      });
    });
  }

  Future<void> _initIfNeeded() async {
    if (_manager.state != VadState.uninitialized) return;
    try {
      await _manager.init();
    } catch (_) {}
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;

    // Dispose previous player.
    await _playerController?.dispose();
    _playerController = null;

    // Create a new video player for the selected file.
    try {
      VideoPlayerController controller;
      if (kIsWeb) {
        // On web, use bytes to create a blob URL.
        final bytes = file.bytes;
        if (bytes == null || bytes.isEmpty) {
          setState(() => _logController.text =
              'Error: Could not read file bytes. Try a smaller file or different browser.');
          return;
        }
        final url = createBlobUrl(bytes);
        setState(() => _logController.text =
            'Loaded: ${file.name} (${(bytes.length / 1024 / 1024).toStringAsFixed(1)} MB)');
        controller = VideoPlayerController.networkUrl(Uri.parse(url));
      } else {
        // On native, use the file path.
        final path = file.path;
        if (path == null) {
          setState(() => _logController.text = 'Error: Could not get file path');
          return;
        }
        controller = VideoPlayerController.networkUrl(Uri.file(path));
      }
      await controller.initialize();

      controller.addListener(() {
        if (!mounted) return;
        setState(() {
          _currentPosition = controller.value.position;
          _totalDuration = controller.value.duration;
          _isPlaying = controller.value.isPlaying;
        });
      });

      setState(() {
        _playerController = controller;
        _fileName = file.name;
        _fileBytes = file.bytes;
        _segments = [];
        _elapsed = 0.0;
        _audioDuration = 0.0;
        _currentPosition = Duration.zero;
        _totalDuration = controller.value.duration;
        _logController.text = 'Loaded: ${file.name}';
      });
    } catch (e) {
      setState(() => _logController.text = 'Error loading file: $e');
    }
  }

  void _togglePlayback() {
    if (_playerController == null) return;
    if (_playerController!.value.isPlaying) {
      _playerController!.pause();
    } else {
      _playerController!.play();
    }
  }

  void _cancelVad() {
    _manager.cancel();
    setState(() {
      _isProcessing = false;
      _progress = 0.0;
      _logController.text = 'Cancelled';
    });
  }

  void _seekTo(Duration position) {
    _playerController?.seekTo(position);
  }

  String _formatDuration(Duration d) {
    final m = d.inMinutes;
    final s = d.inSeconds % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toString().padLeft(2, '0')}';
  }

  String _formatTime(double seconds) {
    final m = seconds ~/ 60;
    final s = seconds % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toStringAsFixed(2).padLeft(5, '0')}';
  }

  Future<void> _runVad() async {
    await _initIfNeeded();

    if (_fileBytes == null) {
      setState(() => _logController.text = 'Please load a file first');
      return;
    }

    // Validate parameters.
    final threshold = double.tryParse(_thresholdController.text.trim());
    final minSilence = double.tryParse(_minSilenceController.text.trim());
    final minSpeech = double.tryParse(_minSpeechController.text.trim());
    final maxSpeech = double.tryParse(_maxSpeechController.text.trim());

    if (threshold == null || threshold < 0.0 || threshold > 1.0) {
      setState(() => _logController.text = 'Error: Threshold must be between 0.0 and 1.0');
      return;
    }
    if (minSilence == null || minSilence <= 0) {
      setState(() => _logController.text = 'Error: Min silence duration must be > 0');
      return;
    }
    if (minSpeech == null || minSpeech <= 0) {
      setState(() => _logController.text = 'Error: Min speech duration must be > 0');
      return;
    }
    if (maxSpeech == null || maxSpeech <= 0) {
      setState(() => _logController.text = 'Error: Max speech duration must be > 0');
      return;
    }
    if (maxSpeech <= minSpeech) {
      setState(() => _logController.text = 'Error: Max speech duration must be > min speech duration');
      return;
    }

    setState(() {
      _isProcessing = true;
      _progress = 0.0;
      _segments = [];
      _logController.text = 'Decoding audio...';
    });

    // Decode the audio file to 16kHz mono PCM.
    final decoded = await decodeAudioBytes(_fileBytes!);
    if (decoded == null) {
      setState(() {
        _isProcessing = false;
        _logController.text = 'Error: Could not decode audio file';
      });
      return;
    }

    setState(() {
      _loadedSamples = decoded.samples;
      _loadedSampleRate = decoded.sampleRate;
      _logController.text =
          'Decoded: ${decoded.duration.toStringAsFixed(2)}s, '
          '${decoded.sampleRate} Hz, '
          '${decoded.samples.length} samples. Running VAD...';
    });

    // Run VAD on the decoded samples.
    _manager.runVad(
      samples: decoded.samples,
      sampleRate: decoded.sampleRate,
      threshold: threshold,
      minSilenceDuration: minSilence,
      minSpeechDuration: minSpeech,
      maxSpeechDuration: maxSpeech,
    );
  }

  // End position for the currently playing segment (0 = no segment limit).
  int _segmentEndMs = 0;
  // Whether we're listening for segment-end position.
  bool _hasSegmentListener = false;

  void _onPlayerPositionUpdate() {
    if (_playerController == null) return;
    final pos = _playerController!.value.position.inMilliseconds;

    // Check if current segment ended.
    if (_segmentEndMs > 0 && pos >= _segmentEndMs) {
      if (_playAllFromSegment && _playingIndex + 1 < _segments.length) {
        // Play the next segment (skip silence).
        _playSegment(_playingIndex + 1);
      } else {
        // Stop playback.
        _playerController!.pause();
        _segmentEndMs = 0;
        if (mounted) setState(() => _playingIndex = -1);
      }
    }
  }

  Future<void> _playSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;

    // If the same segment is already playing, stop it.
    if (_playingIndex == index) {
      _playerController?.pause();
      _segmentEndMs = 0;
      setState(() => _playingIndex = -1);
      return;
    }

    final seg = _segments[index];
    final startMs = (seg.start * 1000).toInt();
    final endMs = (seg.end * 1000).toInt();

    setState(() => _playingIndex = index);

    // Add listener once to check segment end.
    if (!_hasSegmentListener) {
      _playerController?.addListener(_onPlayerPositionUpdate);
      _hasSegmentListener = true;
    }
    _segmentEndMs = endMs;

    // Seek to segment start and play.
    await _playerController?.seekTo(Duration(milliseconds: startMs));
    _playerController?.play();
  }

  Future<void> _saveSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;
    final seg = _segments[index];
    final wavBytes = encodeWav(seg.samples, _loadedSampleRate);
    final startStr = seg.start.toStringAsFixed(2);
    final endStr = seg.end.toStringAsFixed(2);
    // Use the original filename (without extension) as prefix.
    final baseName = _fileName != null
        ? _fileName!.replaceAll(RegExp(r'\.[^.]+$'), '')
        : 'audio';
    final suggestedName =
        '${baseName}_segment_${(index + 1).toString().padLeft(3, '0')}_${startStr}s_${endStr}s.wav';

    if (kIsWeb) {
      await web_audio.saveWavBytesWithDialog(wavBytes, suggestedName);
    } else {
      final savedPath = await save_file.saveWavBytes(wavBytes, suggestedName);
      if (savedPath != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to $savedPath')),
        );
      }
    }
  }

  Future<void> _saveAllSegments() async {
    if (_segments.isEmpty) return;

    // Concatenate all segment samples (silence removed).
    int totalSamples = 0;
    for (final seg in _segments) {
      totalSamples += seg.samples.length;
    }
    final merged = Float32List(totalSamples);
    int offset = 0;
    for (final seg in _segments) {
      merged.setRange(offset, offset + seg.samples.length, seg.samples);
      offset += seg.samples.length;
    }

    final wavBytes = encodeWav(merged, _loadedSampleRate);
    final baseName = _fileName != null
        ? _fileName!.replaceAll(RegExp(r'\.[^.]+$'), '')
        : 'audio';
    final suggestedName = '${baseName}_no_silence.wav';

    if (kIsWeb) {
      await web_audio.saveWavBytesWithDialog(wavBytes, suggestedName);
    } else {
      final savedPath = await save_file.saveWavBytes(wavBytes, suggestedName);
      if (savedPath != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to $savedPath')),
        );
      }
    }
  }

  void _clearAll() {
    _playerController?.pause();

    // Clean up cached WAV files from temp directory.
    cleanupTempChunkFiles();

    setState(() {
      _segments = [];
      _elapsed = 0.0;
      _audioDuration = 0.0;
      _loadedSamples = null;
      _loadedSampleRate = 16000;
      _fileName = null;
      _fileBytes = null;
      _playingIndex = -1;
      _segmentEndMs = 0;
      _playAllFromSegment = false;
      _currentPosition = Duration.zero;
      _totalDuration = Duration.zero;
      _isPlaying = false;
      _logController.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('VAD from File')),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            VadControls(
              thresholdController: _thresholdController,
              minSilenceController: _minSilenceController,
              minSpeechController: _minSpeechController,
              maxSpeechController: _maxSpeechController,
            ),
            const SizedBox(height: 8),
            // File picker and controls.
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                OutlinedButton(
                  onPressed: _pickFile,
                  child: const Text('Pick File'),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _playerController == null ? null : _togglePlayback,
                  child: Text(_isPlaying ? 'Pause' : 'Play'),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _isProcessing ? _cancelVad : _runVad,
                  child: Text(_isProcessing ? 'Cancel' : 'Run VAD'),
                ),
                const SizedBox(width: 8),
                OutlinedButton(
                  onPressed: _clearAll,
                  child: const Text('Clear'),
                ),
              ],
            ),
            // Filename display.
            if (_fileName != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(
                  _fileName!,
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
            const SizedBox(height: 8),
            // Playback slider with position.
            if (_playerController != null && _totalDuration.inSeconds > 0)
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 8),
                child: Row(
                  children: [
                    Text(_formatDuration(_currentPosition),
                        style: const TextStyle(fontSize: 12)),
                    Expanded(
                      child: Slider(
                        value: _currentPosition.inMilliseconds.toDouble(),
                        min: 0,
                        max: _totalDuration.inMilliseconds.toDouble().clamp(1, double.infinity),
                        onChanged: (v) => _seekTo(Duration(milliseconds: v.toInt())),
                      ),
                    ),
                    Text(_formatDuration(_totalDuration),
                        style: const TextStyle(fontSize: 12)),
                  ],
                ),
              ),
            const SizedBox(height: 4),
            // Progress bar during VAD processing.
            if (_isProcessing)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 8),
                child: Column(
                  children: [
                    LinearProgressIndicator(value: _progress),
                    const SizedBox(height: 4),
                    Text(
                        'Processing: ${(_progress * 100).toStringAsFixed(0)}%'),
                  ],
                ),
              ),
            // Stats after processing.
            if (!_isProcessing && _segments.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Text(
                  'Duration: ${_audioDuration.toStringAsFixed(2)}s | '
                  'Elapsed: ${_elapsed.toStringAsFixed(2)}s | '
                  'RTF: ${(_elapsed / _audioDuration).toStringAsFixed(3)} '
                  '(${_elapsed.toStringAsFixed(2)} / ${_audioDuration.toStringAsFixed(2)})',
                  style: const TextStyle(fontSize: 13),
                ),
              ),
            const SizedBox(height: 4),
            // Status log.
            TextField(
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: 'Status',
                isDense: true,
                contentPadding:
                    EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              ),
              maxLines: 2,
              controller: _logController,
              readOnly: true,
            ),
            const SizedBox(height: 4),
            // Segment list.
            if (_segments.isNotEmpty)
              Expanded(
                child: Card(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Padding(
                        padding: const EdgeInsets.all(8),
                        child: Row(
                          children: [
                            Text(
                              () {
                                final speechDuration = _segments.fold<double>(
                                    0, (sum, s) => sum + (s.end - s.start));
                                final kept = _audioDuration > 0
                                    ? (speechDuration / _audioDuration * 100)
                                    : 0.0;
                                final removed = 100 - kept;
                                return 'Segments: ${_segments.length} '
                                    '(${speechDuration.toStringAsFixed(2)}s) '
                                    '| Kept: ${kept.toStringAsFixed(1)}% '
                                    '| Removed: ${removed.toStringAsFixed(1)}%';
                              }(),
                              style: const TextStyle(
                                  fontWeight: FontWeight.bold, fontSize: 13),
                            ),
                            const Spacer(),
                            TextButton(
                              onPressed: () {
                                setState(() {
                                  _segments = [];
                                  _elapsed = 0.0;
                                  _audioDuration = 0.0;
                                });
                              },
                              child: const Text('Clear Results',
                                  style: TextStyle(fontSize: 12)),
                            ),
                            TextButton(
                              onPressed: _saveAllSegments,
                              child: const Text('Save All',
                                  style: TextStyle(fontSize: 12)),
                            ),
                            const SizedBox(width: 4),
                            Tooltip(
                              message: _playAllFromSegment
                                  ? 'Playing all from segment'
                                  : 'Playing single segment',
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Checkbox(
                                    value: _playAllFromSegment,
                                    onChanged: (v) {
                                      setState(() =>
                                          _playAllFromSegment = v ?? false);
                                    },
                                    visualDensity: VisualDensity.compact,
                                    materialTapTargetSize:
                                        MaterialTapTargetSize.shrinkWrap,
                                  ),
                                  const Text('Play all',
                                      style: TextStyle(fontSize: 11)),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),
                      Expanded(
                        child: ListView.builder(
                          itemCount: _segments.length,
                          itemBuilder: (context, index) {
                            final seg = _segments[index];
                            final duration = seg.end - seg.start;
                            final isPlaying = _playingIndex == index;
                            return Container(
                              color: isPlaying
                                  ? Theme.of(context)
                                      .colorScheme
                                      .primaryContainer
                                      .withOpacity(0.4)
                                  : null,
                              child: ListTile(
                              dense: true,
                              leading: IconButton(
                                icon: Icon(
                                  isPlaying ? Icons.stop : Icons.play_arrow,
                                  size: 20,
                                ),
                                onPressed: () => _playSegment(index),
                              ),
                              trailing: IconButton(
                                icon: const Icon(Icons.save, size: 18),
                                onPressed: () => _saveSegment(index),
                              ),
                              title: Text(
                                '#${index + 1}  '
                                '${_formatTime(seg.start)} → '
                                '${_formatTime(seg.end)}  '
                                '(${duration.toStringAsFixed(2)}s)',
                                style: TextStyle(
                                  fontSize: 13,
                                  fontWeight: isPlaying ? FontWeight.bold : FontWeight.normal,
                                ),
                              ),
                            ),
                            );
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _manager.dispose();
    _playerController?.dispose();
    _logController.dispose();
    _thresholdController.dispose();
    _minSilenceController.dispose();
    _minSpeechController.dispose();
    _maxSpeechController.dispose();
    super.dispose();
  }
}
