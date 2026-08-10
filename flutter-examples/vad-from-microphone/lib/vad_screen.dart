// Copyright (c)  2026  Xiaomi Corporation
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:record/record.dart';

import './model_config.dart' as cfg;
import './play_bytes.dart' if (dart.library.js_interop) './play_bytes_stub.dart'
    as play_bytes;
import './save_file.dart' if (dart.library.js_interop) './save_file_stub.dart'
    as save_file;
import './vad_controls.dart';
import './vad_manager.dart';
import './wav_encoder.dart';
import './web_audio.dart' if (dart.library.io) './web_audio_stub.dart'
    as web_audio;

class VadScreen extends StatefulWidget {
  const VadScreen({super.key});

  @override
  State<VadScreen> createState() => _VadScreenState();
}

class _VadScreenState extends State<VadScreen> {
  final _logController = TextEditingController();
  final _thresholdController = TextEditingController(text: '0.8');
  final _minSilenceController = TextEditingController(text: '0.2');
  final _minSpeechController = TextEditingController(text: '0.2');
  final _maxSpeechController = TextEditingController(text: '12.0');

  VadMicManager? _manager;
  AudioRecorder? _recorder;
  AudioPlayer? _player;

  bool _isListening = false;
  bool _isSpeaking = false;
  bool _isPlayingRecording = false;
  int _segmentCount = 0;
  int _playingIndex = -1;

  // Detected segments from VAD.
  final List<VadSegment> _segments = [];

  // Simple sample buffer for feeding exact window-size chunks to VAD.
  final List<double> _sampleBuffer = [];
  int _windowSize = 512; // Updated based on selected model.

  // Microphone selection.
  List<InputDevice> _devices = [];
  InputDevice? _selectedDevice;

  // VAD always uses 16kHz mono.

  // Captured audio for playback after stopping.
  final List<Float32List> _capturedChunks = [];
  int _capturedSamples = 0;
  Uint8List? _recordedWavBytes;

  @override
  void initState() {
    super.initState();
    _recorder = AudioRecorder();
    _player = AudioPlayer();
    _player!.onPlayerComplete.listen((_) {
      if (mounted) {
        setState(() {
          _isPlayingRecording = false;
          _playingIndex = -1;
        });
      }
    });
    _loadDevices();
  }

  Future<void> _loadDevices() async {
    try {
      // On web, listInputDevices() requires microphone permission first.
      // Request permission to ensure devices are visible.
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

  void _togglePlayback() {
    if (_recordedWavBytes == null) return;
    if (_isPlayingRecording) {
      _player?.stop();
      setState(() => _isPlayingRecording = false);
    } else {
      setState(() => _isPlayingRecording = true);
      if (kIsWeb) {
        web_audio.playWavBytes(_recordedWavBytes!);
      } else {
        play_bytes.playWavBytes(_player!, _recordedWavBytes!);
      }
    }
  }

  Future<void> _playSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;

    // If the same segment is already playing, stop it.
    if (_playingIndex == index) {
      _player?.stop();
      setState(() => _playingIndex = -1);
      return;
    }

    final seg = _segments[index];
    final wavBytes = encodeWav(seg.samples, 16000);
    setState(() => _playingIndex = index);

    if (kIsWeb) {
      web_audio.playWavBytes(wavBytes);
      // Reset after estimated duration.
      final duration = seg.samples.length / 16000.0;
      Future.delayed(Duration(milliseconds: (duration * 1000).ceil() + 200), () {
        if (mounted && _playingIndex == index) {
          setState(() => _playingIndex = -1);
        }
      });
    } else {
      await _player?.stop();
      await play_bytes.playWavBytes(_player!, wavBytes);
    }
  }

  Future<void> _saveSegment(int index) async {
    if (index < 0 || index >= _segments.length) return;
    final seg = _segments[index];
    final wavBytes = encodeWav(seg.samples, 16000);
    final startStr = seg.start.toStringAsFixed(2);
    final endStr = seg.end.toStringAsFixed(2);
    final suggestedName =
        'mic_segment_${(index + 1).toString().padLeft(3, '0')}_${startStr}s_${endStr}s.wav';

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

  Future<void> _toggleListening() async {
    if (_isListening) {
      await _stopListening();
    } else {
      await _startListening();
    }
  }

  Future<void> _startListening() async {
    if (!await _recorder!.hasPermission()) {
      setState(() => _logController.text = 'Microphone permission denied');
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

    // Clear captured audio and segments.
    _capturedChunks.clear();
    _capturedSamples = 0;
    _recordedWavBytes = null;
    _segments.clear();
    _playingIndex = -1;

    // Create a fresh VAD manager.
    _manager?.dispose();
    _manager = VadMicManager();

    _manager!.logStream.listen((msg) {
      if (mounted) setState(() => _logController.text = msg);
    });
    _manager!.speechStream.listen((isSpeaking) {
      if (mounted) setState(() => _isSpeaking = isSpeaking);
    });
    _manager!.segmentCountStream.listen((count) {
      if (mounted) setState(() => _segmentCount = count);
    });
    _manager!.segmentsStream.listen((seg) {
      if (mounted) setState(() => _segments.add(seg));
    });

    try {
      await _manager!.init(
        threshold: threshold,
        minSilenceDuration: minSilence,
        minSpeechDuration: minSpeech,
        maxSpeechDuration: maxSpeech,
      );
    } catch (e) {
      setState(() => _logController.text = 'Error: $e');
      return;
    }

    // Set window size based on model.
    _windowSize = cfg.windowSize;
    _sampleBuffer.clear();

    // Start recording with streaming (always mono, 16kHz).
    const encoder = AudioEncoder.pcm16bits;

    // Check encoder support.
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
        _segmentCount = 0;
        _isSpeaking = false;
        _logController.text = 'Recording: 16kHz, mono'
            '${_selectedDevice != null ? ", ${_selectedDevice!.label}" : ""}';
      });

      // Listen for PCM chunks, buffer, and feed exact window-size chunks to VAD.
      stream.listen(
        (Uint8List data) {
          if (!_isListening || _manager == null) return;

          if (data.isEmpty) return;

          // Convert PCM16 bytes to Float32 samples.
          final samples = _convertBytesToFloat32(Uint8List.fromList(data));

          // Store for playback.
          _capturedChunks.add(Float32List.fromList(samples));
          _capturedSamples += samples.length;

          // Add to buffer.
          _sampleBuffer.addAll(samples);

          // Feed window-size chunks to VAD.
          while (_sampleBuffer.length >= _windowSize) {
            final chunk = Float32List.fromList(
                _sampleBuffer.sublist(0, _windowSize));
            _sampleBuffer.removeRange(0, _windowSize);
            _manager!.acceptWaveform(chunk);
          }
        },
        onDone: () {
          if (mounted && _isListening) {
            _stopListening();
          }
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

  /// Convert PCM16 bytes to Float32 samples (same as streaming_asr/utils.dart).
  static Float32List _convertBytesToFloat32(Uint8List bytes) {
    final values = Float32List(bytes.length ~/ 2);
    final data = ByteData.view(bytes.buffer, bytes.offsetInBytes, bytes.lengthInBytes);
    for (var i = 0; i < bytes.length; i += 2) {
      int short = data.getInt16(i, Endian.little);
      values[i ~/ 2] = short / 32768.0;
    }
    return values;
  }

  Future<void> _stopListening() async {
    try {
      await _recorder?.stop();
    } catch (_) {}
    _sampleBuffer.clear();

    // Build WAV bytes from captured chunks for playback.
    if (_capturedChunks.isNotEmpty) {
      final allSamples = Float32List(_capturedSamples);
      int offset = 0;
      for (final chunk in _capturedChunks) {
        allSamples.setRange(offset, offset + chunk.length, chunk);
        offset += chunk.length;
      }
      _recordedWavBytes = encodeWav(allSamples, 16000);
    }

    setState(() {
      _isListening = false;
      _isSpeaking = false;
      _logController.text = 'Stopped. Segments: $_segmentCount';
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Scaffold(
      appBar: AppBar(title: const Text('VAD from Microphone')),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: ListView(
          children: [
            VadControls(
              thresholdController: _thresholdController,
              minSilenceController: _minSilenceController,
              minSpeechController: _minSpeechController,
              maxSpeechController: _maxSpeechController,
            ),
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
                                child: Text(d.label, overflow: TextOverflow.ellipsis),
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
            const SizedBox(height: 16),
            // Circle indicator.
            Center(
              child: Container(
                width: 120,
                height: 120,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: _isSpeaking ? Colors.red : Colors.black,
                  boxShadow: _isSpeaking
                      ? [
                          BoxShadow(
                            color: Colors.red.withOpacity(0.5),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ]
                      : null,
                ),
                child: Center(
                  child: Text(
                    _isSpeaking ? 'SPEECH' : '',
                    style: TextStyle(
                      color: _isSpeaking ? Colors.white : Colors.grey,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
            // Start/Stop button.
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                OutlinedButton(
                  onPressed: _toggleListening,
                  child: Text(_isListening ? 'Stop' : 'Start'),
                ),
                if (_recordedWavBytes != null && !_isListening) ...[
                  const SizedBox(width: 8),
                  OutlinedButton(
                    onPressed: _togglePlayback,
                    child: Text(_isPlayingRecording ? 'Stop Playback' : 'Play Recording'),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 8),
            // Status.
            Center(
              child: Text(
                _isListening ? 'Recording...' : 'Segments: $_segmentCount',
                style: theme.textTheme.bodyLarge,
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
            // Segment list (shown during and after recording).
            if (_segments.isNotEmpty) ...[
              const SizedBox(height: 8),
              Card(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Padding(
                      padding: const EdgeInsets.all(8),
                      child: Row(
                        children: [
                          Text(
                            'Speech Segments (${_segments.length})',
                            style: const TextStyle(
                                fontWeight: FontWeight.bold, fontSize: 14),
                          ),
                          const Spacer(),
                          TextButton(
                            onPressed: () {
                              setState(() {
                                _segments.clear();
                                _segmentCount = 0;
                              });
                            },
                            child: const Text('Clear',
                                style: TextStyle(fontSize: 12)),
                          ),
                        ],
                      ),
                    ),
                    ...List.generate(_segments.length, (index) {
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
                              fontWeight:
                                  isPlaying ? FontWeight.bold : FontWeight.normal,
                            ),
                          ),
                        ),
                      );
                    }),
                  ],
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
