// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';

import './punct_manager.dart';
import './punct_controls.dart';

class PunctScreen extends StatefulWidget {
  const PunctScreen({super.key});

  @override
  State<PunctScreen> createState() => _PunctScreenState();
}

class _PunctScreenState extends State<PunctScreen> {
  final _inputController = TextEditingController();
  final _resultController = TextEditingController();
  final _logController = TextEditingController();

  late final PunctManager _manager;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _manager = PunctManager();

    _manager.logStream.listen((msg) {
      if (mounted) {
        setState(() => _logController.text = msg);
      }
    });

    _manager.resultStream.listen((result) {
      if (!mounted) return;
      setState(() {
        _isProcessing = false;
        _resultController.text = result.result;
        _logController.text =
            'Elapsed: ${result.elapsed.toStringAsFixed(3)}s';
      });
    });
  }

  Future<void> _initIfNeeded() async {
    if (_manager.state != PunctState.uninitialized) return;
    try {
      await _manager.init();
    } catch (_) {}
  }

  Future<void> _onPunctuate() async {
    await _initIfNeeded();

    final text = _inputController.text.trim();
    if (text.isEmpty) {
      setState(() => _logController.text = 'Please enter text');
      return;
    }

    setState(() {
      _isProcessing = true;
      _resultController.clear();
    });
    _manager.punctuate(text);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Online Punctuation')),
      body: Padding(
        padding: const EdgeInsets.all(10),
        child: Column(
          children: [
            PunctControls(
              inputController: _inputController,
              resultController: _resultController,
              onPunctuate: _onPunctuate,
              onClear: () {
                _inputController.clear();
                _resultController.clear();
                _logController.clear();
              },
              isProcessing: _isProcessing,
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
          ],
        ),
      ),
    );
  }

  @override
  void dispose() {
    _manager.dispose();
    _inputController.dispose();
    _resultController.dispose();
    _logController.dispose();
    super.dispose();
  }
}
