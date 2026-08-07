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
    'yesterday afternoon 我去了一家 near my apartment 的 coffee shop '
    '想要 enjoy a cup of hot latte 并 check 一些 important work emails '
    '顺便在我的 laptop 上写一点 code 因为那里的 atmosphere 总是非常 '
    'quiet and comfortable 可以让我更加 focus on coding 和 writing documents '
    '而且没有任何 distractions 所以 if you also like this kind of relaxing '
    'weekend vibe 我们 definitely 应该 plan 一个 time 一起 hang out.';

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
      appBar: AppBar(title: const Text('Offline Punctuation')),
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
