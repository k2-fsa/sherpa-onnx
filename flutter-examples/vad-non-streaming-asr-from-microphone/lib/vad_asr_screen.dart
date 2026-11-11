// Copyright (c)  2026  Xiaomi Corporation
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:record/record.dart';

import './model.dart' if (dart.library.js_interop) './model_web.dart' as model;
import './model_config.dart' as cfg;
import './play_bytes.dart' if (dart.library.js_interop) './play_bytes_stub.dart'
    as play_bytes;
import './save_file.dart' if (dart.library.js_interop) './save_file_stub.dart'
    as save_file;
import './web_audio.dart' if (dart.library.io) './web_audio_stub.dart'
    as web_audio;
import './vad_asr_manager.dart'
    if (dart.library.js_interop) './vad_asr_manager_web.dart';
import './wav_encoder.dart';

class VadAsrScreen extends StatefulWidget {
  const VadAsrScreen({super.key});

  @override
  State<VadAsrScreen> createState() => _VadAsrScreenState();
}

class _VadAsrScreenState extends State<VadAsrScreen> {
  final _logController = TextEditingController();
  final _thresholdController = TextEditingController(text: '0.8');
  final _minSilenceController = TextEditingController(text: '0.2');
  final _minSpeechController = TextEditingController(text: '0.2');
  final _maxSpeechController = TextEditingController(text: '12.0');

  VadAsrManager? _manager;
  AudioRecorder? _recorder;
  AudioPlayer? _player;

  bool _isListening = false;
  bool _isSpeaking = false;
  int _playingIndex = -1;

  final List<VadAsrSegment> _segments = [];
  final Set<int> _editingIndices = {};

  // Sample buffer for feeding exact window-size chunks to VAD.
  final List<double> _sampleBuffer = [];
  int _windowSize = 512;

  // Microphone selection.
  List<InputDevice> _devices = [];
  InputDevice? _selectedDevice;

  @override
  void initState() {
    super.initState();
    _recorder = AudioRecorder();
    _player = AudioPlayer();
    _loadDevices();
  }

  Future<void> _loadDevices() async {
    try {
      await _recorder!.hasPermission();
      final devices = await _recorder!.listInputDevices();
      setState(() {
        _devices = devices;
        if (devices.isNotEmpty && _selectedDevice == null) {
          _selectedDevice = devices.first;
        }
      });
    } catch (e) {
      setState(() => _logController.text = 'Error listing devices: $e');
    }
  }

  Future<bool> _initManager({
    required double threshold,
    required double minSilenceDuration,
    required double minSpeechDuration,
    required double maxSpeechDuration,
  }) async {
    // Create manager once — ASR recognizer stays alive across restarts.
    if (_manager == null) {
      _manager = VadAsrManager();

      _manager!.logStream.listen((msg) {
        if (mounted) setState(() => _logController.text = msg);
      });

      _manager!.segmentStream.listen((seg) {
        if (!mounted) return;
        setState(() {
          _segments.add(seg);
          _logController.text =
              'Segment #${_segments.length} detected, decoding...';
        });
      });
      _manager!.textUpdateStream.listen((update) {
        if (!mounted) return;
        if (update.index >= 0 && update.index < _segments.length) {
          final seg = _segments[update.index];
          final audioDur = seg.end - seg.start;
          final rtf = audioDur > 0 ? update.elapsedSeconds / audioDur : 0.0;
          setState(() {
            seg.text = update.text;
            seg.elapsedSeconds = update.elapsedSeconds;
            if (!_isSpeaking) {
              _logController.text =
                  'Segment #${update.index + 1}: ${update.text}\n'
                  'RTF = ${update.elapsedSeconds.toStringAsFixed(3)}/${audioDur.toStringAsFixed(2)} = ${rtf.toStringAsFixed(3)}';
            }
          });
        }
      });

      _manager!.speechStream.listen((isSpeech) {
        if (!mounted) return;
        setState(() {
          _isSpeaking = isSpeech;
          if (isSpeech) {
            _logController.text = 'Speech detected, listening...';
          } else {
            _logController.text =
                'Silence detected. ${_segments.length} segments so far.';
          }
        });
      });

      try {
        await model.prepareModelConfig();
        final dirs = await model.prepareModelDirs();

        await _manager!.init(
          modelDir: dirs.asrModelDir,
          vadModelDir: dirs.baseDir,
          threshold: threshold,
          minSilenceDuration: minSilenceDuration,
          minSpeechDuration: minSpeechDuration,
          maxSpeechDuration: maxSpeechDuration,
        );
        _windowSize = cfg.defaultVadConfig.sileroVad.windowSize;
      } catch (e) {
        if (mounted) setState(() => _logController.text = 'Init error: $e');
        _manager?.dispose();
        _manager = null;
        return false;
      }
    } else {
      // Manager exists — just reset VAD for a new recording session.
      _manager!.reset();
    }
    return true;
  }

  Future<void> _startListening() async {
    final threshold = double.tryParse(_thresholdController.text.trim()) ?? 0.8;
    final minSilence = double.tryParse(_minSilenceController.text.trim()) ?? 0.2;
    final minSpeech = double.tryParse(_minSpeechController.text.trim()) ?? 0.2;
    final maxSpeech = double.tryParse(_maxSpeechController.text.trim()) ?? 12.0;

    if (!await _initManager(
      threshold: threshold,
      minSilenceDuration: minSilence,
      minSpeechDuration: minSpeech,
      maxSpeechDuration: maxSpeech,
    )) return;

    if (!await _recorder!.hasPermission()) {
      setState(() => _logController.text = 'Microphone permission denied');
      return;
    }

    _sampleBuffer.clear();

    const encoder = AudioEncoder.pcm16bits;
    if (!await _recorder!.isEncoderSupported(encoder)) {
      setState(() => _logController.text = 'Error: PCM16 encoder not supported');
      return;
    }

    final config = RecordConfig(
      encoder: encoder,
      sampleRate: 16000,
      numChannels: 1,
      device: _selectedDevice,
    );

    try {
      final stream = await _recorder!.startStream(config);

      setState(() {
        _isListening = true;
        _segments.clear();
        _editingIndices.clear();
        _logController.text = 'Listening...';
      });

      stream.listen(
        (Uint8List data) {
          if (!_isListening || _manager == null) return;
          if (data.isEmpty) return;

          // Convert PCM16 bytes to Float32 samples.
          final samples = _convertBytesToFloat32(Uint8List.fromList(data));
          _sampleBuffer.addAll(samples);

          // Feed exact window-size chunks to VAD.
          while (_sampleBuffer.length >= _windowSize) {
            final chunk = Float32List.fromList(
                _sampleBuffer.sublist(0, _windowSize));
            _sampleBuffer.removeRange(0, _windowSize);
            _manager?.acceptWaveform(chunk);
          }
        },
        onDone: () {
          if (mounted && _isListening) _stopListening();
        },
        onError: (e) {
          if (mounted) {
            setState(() => _logController.text = 'Stream error: $e');
          }
        },
      );
    } catch (e) {
      setState(() => _logController.text = 'Error starting recording: $e');
    }
  }

  Future<void> _stopListening() async {
    await _recorder!.stop();
    setState(() {
      _isListening = false;
      _isSpeaking = false;
      _logController.text = 'Stopped. ${_segments.length} segments detected.';
    });
  }

  Float32List _convertBytesToFloat32(Uint8List bytes) {
    // PCM16 little-endian to Float32.
    final numSamples = bytes.length ~/ 2;
    final samples = Float32List(numSamples);
    for (int i = 0; i < numSamples; i++) {
      final lo = bytes[i * 2];
      final hi = bytes[i * 2 + 1];
      final val = (hi << 8) | lo;
      // Sign extend 16-bit.
      final signed = val >= 0x8000 ? val - 0x10000 : val;
      samples[i] = signed / 32768.0;
    }
    return samples;
  }

  Future<void> _playSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;
    if (_playingIndex == index) {
      await _player!.stop();
      setState(() => _playingIndex = -1);
      return;
    }

    final seg = _segments[index];
    final wavBytes = encodeWav(seg.samples, 16000);
    setState(() => _playingIndex = index);

    if (kIsWeb) {
      web_audio.playWavBytes(wavBytes);
      final duration = seg.samples.length / 16000.0;
      Future.delayed(
          Duration(milliseconds: (duration * 1000).ceil() + 200), () {
        if (mounted && _playingIndex == index) {
          setState(() => _playingIndex = -1);
        }
      });
    } else {
      try {
        await play_bytes.playWavBytes(_player!, wavBytes);
        _player!.onPlayerComplete.listen((_) {
          if (mounted) setState(() => _playingIndex = -1);
        });
      } catch (e) {
        setState(() => _playingIndex = -1);
      }
    }
  }

  Future<void> _saveSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;
    final seg = _segments[index];
    final wavBytes = encodeWav(seg.samples, 16000);
    final suggestedName =
        'segment_${(index + 1).toString().padLeft(3, '0')}.wav';

    if (kIsWeb) {
      web_audio.saveWavBytesWithDialog(wavBytes, suggestedName);
    } else {
      final savedPath = await save_file.saveWavBytes(wavBytes, suggestedName);
      if (savedPath != null && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved to $savedPath')),
        );
      }
    }
  }

  String _formatTime(double seconds) {
    final m = seconds ~/ 60;
    final s = seconds % 60;
    return '${m.toString().padLeft(2, '0')}:${s.toStringAsFixed(2).padLeft(5, '0')}';
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('VAD + ASR from Microphone')),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            _buildParamControls(),
            const SizedBox(height: 8),
            _buildActionButtons(),
            const SizedBox(height: 8),
            // Microphone settings.
            Card(
              child: Padding(
                padding: const EdgeInsets.all(12),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text('Microphone Settings',
                        style: TextStyle(fontWeight: FontWeight.bold)),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        const Text('Device: '),
                        Expanded(
                          child: DropdownButton<InputDevice>(
                            isExpanded: true,
                            value: _selectedDevice,
                            items: _devices.map((d) {
                              return DropdownMenuItem(
                                value: d,
                                child: Text(d.label,
                                    overflow: TextOverflow.ellipsis),
                              );
                            }).toList(),
                            onChanged: _isListening
                                ? null
                                : (v) => setState(() => _selectedDevice = v),
                          ),
                        ),
                        IconButton(
                          icon: const Icon(Icons.refresh, size: 18),
                          tooltip: 'Refresh devices',
                          onPressed: _isListening ? null : _loadDevices,
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    const Text('Recording: 16kHz mono (required by VAD)',
                        style: TextStyle(fontSize: 13, color: Colors.grey)),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 8),
            if (_isSpeaking)
              const Padding(
                padding: EdgeInsets.symmetric(vertical: 4),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.mic, color: Colors.red),
                    SizedBox(width: 8),
                    Text('Speaking...', style: TextStyle(color: Colors.red)),
                  ],
                ),
              ),
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
                        padding: const EdgeInsets.all(8),
                        child: Row(
                          children: [
                            Text(
                              'Segments: ${_segments.length}',
                              style: const TextStyle(
                                  fontWeight: FontWeight.bold, fontSize: 13),
                            ),
                            const SizedBox(width: 8),
                            Tooltip(
                              message: 'RTF = ASR elapsed time / audio duration\n'
                                  'RTF < 1.0 means faster than real-time',
                              child: const Icon(Icons.info_outline, size: 16),
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
                            final rtfStr = seg.elapsedSeconds > 0
                                ? ' RTF=${seg.elapsedSeconds.toStringAsFixed(3)}/${duration.toStringAsFixed(2)}=${(seg.elapsedSeconds / duration).toStringAsFixed(3)}'
                                : '';
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
                                  '(${duration.toStringAsFixed(2)}s$rtfStr)',
                                  style: TextStyle(
                                    fontSize: 13,
                                    fontWeight: isPlaying
                                        ? FontWeight.bold
                                        : FontWeight.normal,
                                  ),
                                ),
                                subtitle: isEditing
                                    ? _EditingTextField(
                                        initialText: seg.text,
                                        onSaved: (value) {
                                          setState(() {
                                            seg.text = value;
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
                                        child: seg.text.isNotEmpty
                                            ? Text(seg.text,
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
          onPressed: _isListening ? _stopListening : _startListening,
          child: Text(_isListening ? 'Stop' : 'Start'),
        ),
        const SizedBox(width: 8),
        OutlinedButton(
          onPressed: _isListening
              ? null
              : () {
                  _manager?.reset();
                  setState(() {
                    _segments.clear();
                    _editingIndices.clear();
                    _isSpeaking = false;
                    _logController.text = 'Cleared';
                  });
                },
          child: const Text('Clear'),
        ),
      ],
    );
  }

  @override
  void dispose() {
    _manager?.dispose();
    _recorder?.dispose();
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
