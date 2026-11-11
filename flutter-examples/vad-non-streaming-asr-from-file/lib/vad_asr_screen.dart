// Copyright (c)  2026  Xiaomi Corporation
import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:media_kit/media_kit.dart';

import './audio_decoder_native.dart'
    if (dart.library.js_interop) './audio_decoder_web.dart';
import './model.dart' if (dart.library.js_interop) './model_web.dart' as model;
import './model_config.dart' as cfg;
import './vad_asr_manager.dart'
    if (dart.library.js_interop) './vad_asr_manager_web.dart';
import './wav_encoder.dart';
import './save_file.dart' if (dart.library.js_interop) './save_file_stub.dart'
    as save_file;
import './web_blob_url.dart' if (dart.library.io) './web_blob_url_stub.dart';

class VadAsrScreen extends StatefulWidget {
  const VadAsrScreen({super.key});

  @override
  State<VadAsrScreen> createState() => _VadAsrScreenState();
}

class _VadAsrScreenState extends State<VadAsrScreen> {
  final _logController = TextEditingController();
  final _thresholdController = TextEditingController(text: '0.1');
  final _minSilenceController = TextEditingController(text: '0.5');
  final _minSpeechController = TextEditingController(text: '0.25');
  final _maxSpeechController = TextEditingController(text: '12.0');

  late final VadAsrManager _manager;
  Player? _player;

  List<VadAsrSegment> _segments = [];
  bool _isProcessing = false;
  double _progress = 0.0;
  double _elapsed = 0.0;
  double _audioDuration = 0.0;
  String? _fileName;
  Uint8List? _fileBytes;
  int _playingIndex = -1;
  bool _autoPlayNext = false;
  StreamSubscription<Duration>? _positionSub;

  // Editable text support: index → edited text.
  final Map<int, String> _editedTexts = {};
  final Set<int> _editingIndices = {};

  Duration _currentPosition = Duration.zero;
  Duration _totalDuration = Duration.zero;
  bool _isPlaying = false;

  @override
  void initState() {
    super.initState();
    _manager = VadAsrManager();

    _manager.logStream.listen((msg) {
      if (mounted) setState(() => _logController.text = msg);
    });

    _manager.progressStream.listen((progress) {
      if (mounted) setState(() => _progress = progress);
    });

    _manager.segmentStream.listen((seg) {
      if (!mounted) return;
      setState(() {
        _segments.add(seg);
        _logController.text = 'Processing... ${_segments.length} segments found';
      });
    });

    _manager.resultStream.listen((result) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _progress = 0.0;
        _elapsed = result.elapsed;
        _audioDuration = result.audioDuration;
        final rtf = _audioDuration > 0 ? _elapsed / _audioDuration : 0.0;
        _logController.text =
            'Done! ${_segments.length} segments | '
            '${_audioDuration.toStringAsFixed(1)}s audio | '
            '${_elapsed.toStringAsFixed(1)}s elapsed | '
            'RTF ${rtf.toStringAsFixed(3)}';
      });
    });
  }

  Future<bool> _initIfNeeded() async {
    if (_manager.state != VadAsrState.uninitialized) return true;
    try {
      // Copy model assets to disk and get resolved paths.
      await model.prepareModelConfig();
      final dirs = await model.prepareModelDirs();
      await _manager.init(modelDir: dirs.asrModelDir, vadModelDir: dirs.baseDir);
      return true;
    } catch (e) {
      setState(() => _logController.text = 'Init error: $e');
      return false;
    }
  }

  Future<void> _pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.any,
      withData: true,
    );
    if (result == null || result.files.isEmpty) return;

    final file = result.files.first;

    await _player?.dispose();
    _player = null;

    try {
      if (kIsWeb) {
        final bytes = file.bytes;
        if (bytes == null || bytes.isEmpty) {
          setState(() => _logController.text =
              'Error: Could not read file bytes.');
          return;
        }
        final url = createBlobUrl(bytes);
        _player = Player();
        await _player!.open(Media(url), play: false);

        _player!.stream.position.listen((pos) {
          if (!mounted) return;
          setState(() => _currentPosition = pos);
        });
        _player!.stream.duration.listen((dur) {
          if (!mounted) return;
          setState(() => _totalDuration = dur);
        });
        _player!.stream.playing.listen((playing) {
          if (!mounted) return;
          setState(() => _isPlaying = playing);
        });

        setState(() {
          _fileName = file.name;
          _fileBytes = file.bytes;
          _currentPosition = Duration.zero;
          _totalDuration = _player!.state.duration;
          _logController.text = 'Loaded: ${file.name}';
        });
      } else {
        final path = file.path;
        if (path == null) {
          setState(() => _logController.text = 'Error: Could not get file path');
          return;
        }
        _player = Player();
        await _player!.open(Media(path), play: false);

        _player!.stream.position.listen((pos) {
          if (!mounted) return;
          setState(() => _currentPosition = pos);
        });
        _player!.stream.duration.listen((dur) {
          if (!mounted) return;
          setState(() => _totalDuration = dur);
        });
        _player!.stream.playing.listen((playing) {
          if (!mounted) return;
          setState(() => _isPlaying = playing);
        });

        setState(() {
          _fileName = file.name;
          _fileBytes = file.bytes;
          _currentPosition = Duration.zero;
          _totalDuration = _player!.state.duration;
          _logController.text = 'Loaded: ${file.name}';
        });
      }

      setState(() {
        _segments = [];
        _elapsed = 0.0;
        _audioDuration = 0.0;
      });
    } catch (e) {
      setState(() => _logController.text = 'Error loading file: $e');
    }
  }

  void _togglePlayback() {
    if (_player == null) return;
    _player!.playOrPause();
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
    _player?.seek(position);
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

  Future<void> _runVadAsr() async {
    if (!await _initIfNeeded()) return;

    if (_fileBytes == null) {
      setState(() => _logController.text = 'Please load a file first');
      return;
    }

    final threshold = double.tryParse(_thresholdController.text.trim());
    final minSilence = double.tryParse(_minSilenceController.text.trim());
    final minSpeech = double.tryParse(_minSpeechController.text.trim());
    final maxSpeech = double.tryParse(_maxSpeechController.text.trim());

    if (threshold == null || threshold < 0.0 || threshold > 1.0) {
      setState(() => _logController.text = 'Error: Threshold must be 0.0-1.0');
      return;
    }
    if (minSilence == null || minSilence <= 0) {
      setState(() => _logController.text = 'Error: Min silence must be > 0');
      return;
    }
    if (minSpeech == null || minSpeech <= 0) {
      setState(() => _logController.text = 'Error: Min speech must be > 0');
      return;
    }
    if (maxSpeech == null || maxSpeech <= 0) {
      setState(() => _logController.text = 'Error: Max speech must be > 0');
      return;
    }
    if (maxSpeech <= minSpeech) {
      setState(() => _logController.text =
          'Error: Max speech must be > min speech');
      return;
    }

    setState(() {
      _isProcessing = true;
      _progress = 0.0;
      _segments = [];
      _editedTexts.clear();
      _editingIndices.clear();
      _logController.text = kIsWeb
          ? 'Decoding audio... (ASR on web can be slow with large models)'
          : 'Decoding audio...';
    });

    final decoded = await decodeAudioBytes(_fileBytes!);
    if (decoded == null) {
      setState(() {
        _isProcessing = false;
        _logController.text = 'Error: Could not decode audio file';
      });
      return;
    }

    setState(() {
      _logController.text =
          'Decoded: ${decoded.duration.toStringAsFixed(2)}s, '
          '${decoded.sampleRate} Hz. Running VAD + ASR...';
    });

    _manager.runVad(
      samples: decoded.samples,
      sampleRate: decoded.sampleRate,
      threshold: threshold,
      minSilenceDuration: minSilence,
      minSpeechDuration: minSpeech,
      maxSpeechDuration: maxSpeech,
    );
  }

  void _startSegmentPlayback(int index) {
    if (index < 0 || index >= _segments.length || _player == null) return;

    final seg = _segments[index];
    final startMs = (seg.start * 1000).toInt();
    final endMs = (seg.end * 1000).toInt();

    setState(() => _playingIndex = index);
    _positionSub?.cancel();
    _positionSub = _player!.stream.position.listen((pos) {
      if (!mounted) return;
      if (pos.inMilliseconds >= endMs) {
        if (_autoPlayNext && index + 1 < _segments.length) {
          _startSegmentPlayback(index + 1);
        } else {
          _player?.pause();
          _positionSub?.cancel();
          setState(() => _playingIndex = -1);
        }
      }
    });

    _player!.seek(Duration(milliseconds: startMs)).then((_) {
      _player!.play();
    });
  }

  Future<void> _playSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;
    if (_playingIndex == index) {
      _player?.pause();
      _positionSub?.cancel();
      setState(() => _playingIndex = -1);
      return;
    }
    _startSegmentPlayback(index);
  }

  Future<void> _saveSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;
    final seg = _segments[index];
    final wavBytes = encodeWav(seg.samples, 16000);
    final baseName = _fileName != null
        ? _fileName!.replaceAll(RegExp(r'\.[^.]+$'), '')
        : 'audio';
    final suggestedName =
        '${baseName}_segment_${(index + 1).toString().padLeft(3, '0')}.wav';

    final savedPath = await save_file.saveWavBytes(wavBytes, suggestedName);
    if (savedPath != null && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Saved to $savedPath')),
      );
    }
  }

  String _textForSegment(int index) =>
      _editedTexts[index] ?? _segments[index].text;

  /// Format seconds → SRT timestamp "HH:MM:SS,mmm".
  String _srtTimestamp(double seconds) {
    final h = seconds ~/ 3600;
    final m = (seconds % 3600) ~/ 60;
    final s = seconds % 60;
    final whole = s.floor();
    final ms = ((s - whole) * 1000).round();
    return '${h.toString().padLeft(2, '0')}:'
        '${m.toString().padLeft(2, '0')}:'
        '${whole.toString().padLeft(2, '0')},'
        '${ms.toString().padLeft(3, '0')}';
  }

  String _buildSrtContent() {
    final buf = StringBuffer();
    for (int i = 0; i < _segments.length; i++) {
      final seg = _segments[i];
      buf.writeln(i + 1);
      buf.writeln('${_srtTimestamp(seg.start)} --> ${_srtTimestamp(seg.end)}');
      buf.writeln(_textForSegment(i));
      buf.writeln();
    }
    return buf.toString();
  }

  String _buildTxtContent() {
    final buf = StringBuffer();
    for (int i = 0; i < _segments.length; i++) {
      final seg = _segments[i];
      buf.writeln('${_srtTimestamp(seg.start)} --> ${_srtTimestamp(seg.end)}');
      buf.writeln(_textForSegment(i));
      buf.writeln();
    }
    return buf.toString();
  }

  Future<void> _exportSrt() async {
    if (_segments.isEmpty) return;
    final content = _buildSrtContent();
    final baseName = _fileName != null
        ? _fileName!.replaceAll(RegExp(r'\.[^.]+$'), '')
        : 'transcription';
    final suggestedName = '$baseName.srt';

    try {
      final result =
          await save_file.saveTextContent(content, suggestedName, 'srt');
      if (result != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to $result')),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _logController.text = 'Error saving SRT: $e');
      }
    }
  }

  Future<void> _exportTxt() async {
    if (_segments.isEmpty) return;
    final content = _buildTxtContent();
    final baseName = _fileName != null
        ? _fileName!.replaceAll(RegExp(r'\.[^.]+$'), '')
        : 'transcription';
    final suggestedName = '$baseName.txt';

    try {
      final result =
          await save_file.saveTextContent(content, suggestedName, 'txt');
      if (result != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to $result')),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _logController.text = 'Error saving TXT: $e');
      }
    }
  }

  void _clearAll() {
    if (_isProcessing) _cancelVad();
    _player?.pause();
    _positionSub?.cancel();
    setState(() {
      _segments = [];
      _editedTexts.clear();
      _editingIndices.clear();
      _elapsed = 0.0;
      _audioDuration = 0.0;
      _fileName = null;
      _fileBytes = null;
      _playingIndex = -1;
      _currentPosition = Duration.zero;
      _totalDuration = Duration.zero;
      _isPlaying = false;
      _logController.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('VAD + ASR from File')),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            _buildParamControls(),
            const SizedBox(height: 8),
            _buildActionButtons(),
            if (_fileName != null)
              Padding(
                padding: const EdgeInsets.only(top: 4),
                child: Text(_fileName!,
                    style: const TextStyle(fontSize: 12, color: Colors.grey),
                    overflow: TextOverflow.ellipsis),
              ),
            const SizedBox(height: 8),
            if (_player != null && _totalDuration.inSeconds > 0)
              _buildPlaybackSlider(),
            const SizedBox(height: 4),
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
            if (!_isProcessing && _segments.isNotEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 4),
                child: Text(
                  'Duration: ${_audioDuration.toStringAsFixed(2)}s | '
                  'Elapsed: ${_elapsed.toStringAsFixed(2)}s | '
                  'RTF: ${(_elapsed / _audioDuration).toStringAsFixed(3)}',
                  style: const TextStyle(fontSize: 13),
                ),
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
              maxLines: 2,
              controller: _logController,
              readOnly: true,
            ),
            const SizedBox(height: 4),
            if (_segments.isNotEmpty)
              Expanded(
                child: Card(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Padding(
                        padding: const EdgeInsets.symmetric(
                            horizontal: 8, vertical: 4),
                        child: Row(
                          children: [
                            Text(
                              'Segments: ${_segments.length}',
                              style: const TextStyle(
                                  fontWeight: FontWeight.bold, fontSize: 13),
                            ),
                            const SizedBox(width: 8),
                            SizedBox(
                              height: 24,
                              child: Row(
                                mainAxisSize: MainAxisSize.min,
                                children: [
                                  Checkbox(
                                    value: _autoPlayNext,
                                    onChanged: (v) {
                                      setState(() => _autoPlayNext = v ?? false);
                                    },
                                    visualDensity: VisualDensity.compact,
                                    materialTapTargetSize:
                                        MaterialTapTargetSize.shrinkWrap,
                                  ),
                                  const Text('Auto-play',
                                      style: TextStyle(fontSize: 11)),
                                ],
                              ),
                            ),
                            const Spacer(),
                            TextButton(
                              onPressed: _exportSrt,
                              child: const Text('Export SRT',
                                  style: TextStyle(fontSize: 12)),
                            ),
                            TextButton(
                              onPressed: _exportTxt,
                              child: const Text('Export TXT',
                                  style: TextStyle(fontSize: 12)),
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
                            final isEditing = _editingIndices.contains(index);
                            final displayText = _textForSegment(index);

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
                                    fontWeight: isPlaying
                                        ? FontWeight.bold
                                        : FontWeight.normal,
                                  ),
                                ),
                                subtitle: isEditing
                                    ? _EditingTextField(
                                        initialText: displayText,
                                        onSaved: (value) {
                                          setState(() {
                                            _editedTexts[index] = value;
                                            _editingIndices.remove(index);
                                          });
                                        },
                                      )
                                    : GestureDetector(
                                        onDoubleTap: () {
                                          setState(() {
                                            _editingIndices.add(index);
                                          });
                                        },
                                        child: displayText.isNotEmpty
                                            ? Text(displayText,
                                                style: const TextStyle(
                                                    fontSize: 12))
                                            : null,
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

  Widget _buildParamControls() {
    return ExpansionTile(
      title: const Text('VAD Parameters', style: TextStyle(fontSize: 14)),
      initiallyExpanded: false,
      children: [
        Row(
          children: [
            _paramField('Threshold', _thresholdController),
            _paramField('Min Silence', _minSilenceController),
          ],
        ),
        Row(
          children: [
            _paramField('Min Speech', _minSpeechController),
            _paramField('Max Speech', _maxSpeechController),
          ],
        ),
      ],
    );
  }

  Widget _paramField(String label, TextEditingController controller) {
    return Expanded(
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 2),
        child: TextField(
          decoration: InputDecoration(
            labelText: label,
            isDense: true,
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 8, vertical: 8),
            border:
                OutlineInputBorder(borderRadius: BorderRadius.circular(6)),
          ),
          controller: controller,
          keyboardType: const TextInputType.numberWithOptions(decimal: true),
        ),
      ),
    );
  }

  Widget _buildActionButtons() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        OutlinedButton(
          onPressed: _pickFile,
          child: const Text('Pick File'),
        ),
        const SizedBox(width: 8),
        OutlinedButton(
          onPressed: _player == null ? null : _togglePlayback,
          child: Text(_isPlaying ? 'Pause' : 'Play'),
        ),
        const SizedBox(width: 8),
        OutlinedButton(
          onPressed: _isProcessing ? _cancelVad : _runVadAsr,
          child: Text(_isProcessing ? 'Cancel' : 'Run VAD+ASR'),
        ),
        const SizedBox(width: 8),
        OutlinedButton(
          onPressed: _isProcessing ? null : _clearAll,
          child: const Text('Clear'),
        ),
      ],
    );
  }

  Widget _buildPlaybackSlider() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 8),
      child: Row(
        children: [
          Text(_formatDuration(_currentPosition),
              style: const TextStyle(fontSize: 12)),
          Expanded(
            child: Slider(
              value: _currentPosition.inMilliseconds.toDouble(),
              min: 0,
              max: _totalDuration.inMilliseconds
                  .toDouble()
                  .clamp(1, double.infinity),
              onChanged: (v) =>
                  _seekTo(Duration(milliseconds: v.toInt())),
            ),
          ),
          Text(_formatDuration(_totalDuration),
              style: const TextStyle(fontSize: 12)),
        ],
      ),
    );
  }

  @override
  void dispose() {
    _positionSub?.cancel();
    _manager.dispose();
    _player?.dispose();
    _logController.dispose();
    _thresholdController.dispose();
    _minSilenceController.dispose();
    _minSpeechController.dispose();
    _maxSpeechController.dispose();
    super.dispose();
  }
}

/// A text field that saves on submit and on focus loss.
class _EditingTextField extends StatefulWidget {
  final String initialText;
  final ValueChanged<String> onSaved;

  const _EditingTextField({required this.initialText, required this.onSaved});

  @override
  State<_EditingTextField> createState() => _EditingTextFieldState();
}

class _EditingTextFieldState extends State<_EditingTextField> {
  late final TextEditingController _controller;
  late final FocusNode _focusNode;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.initialText);
    _focusNode = FocusNode();
    _focusNode.addListener(() {
      if (!_focusNode.hasFocus) {
        widget.onSaved(_controller.text);
      }
    });
  }

  @override
  void dispose() {
    _controller.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return TextField(
      controller: _controller,
      focusNode: _focusNode,
      autofocus: true,
      style: const TextStyle(fontSize: 12),
      maxLines: null,
      decoration: const InputDecoration(
        isDense: true,
        contentPadding: EdgeInsets.symmetric(horizontal: 4, vertical: 4),
        border: OutlineInputBorder(),
      ),
      onSubmitted: (value) {
        widget.onSaved(value);
      },
    );
  }
}
