// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';

import './punct_manager.dart';
import './punct_controls.dart';

class PunctScreen extends StatefulWidget {
  const PunctScreen({super.key});

  @override
  State<PunctScreen> createState() => _PunctScreenState();
}

const _exampleText =
    'last week my friends and i decided to take a spontaneous road trip down '
    'the coast we packed our bags early on saturday morning and hit the highway '
    'before the sun was even up the weather was absolutely perfect with clear '
    'blue skies and a gentle breeze blowing through the open windows as we '
    'drove we played our favorite playlist and sang along at the top of our '
    'lungs around noon we stopped at a small coastal town to grab some lunch '
    'we found a cozy little diner right by the pier where we ordered fresh '
    'fish and chips and homemade lemonade after eating we walked along the '
    'beach collected a few unique seashells and took tons of pictures to '
    'capture the memory it was honestly one of the best weekends ive had in a '
    'long time and i cannot wait until our next adventure';

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
      final wordCount = _inputController.text.trim().split(RegExp(r'\s+')).length;
      setState(() {
        _isProcessing = false;
        _resultController.text = result.result;
        _logController.text =
            'Words: $wordCount | Elapsed: ${result.elapsed.toStringAsFixed(3)}s';
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
              onExample: () {
                _inputController.text = _exampleText;
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
