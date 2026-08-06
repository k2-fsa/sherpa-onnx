// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';

/// Input/output controls for punctuation.
class PunctControls extends StatelessWidget {
  final TextEditingController inputController;
  final TextEditingController resultController;
  final VoidCallback onPunctuate;
  final VoidCallback onClear;
  final bool isProcessing;

  const PunctControls({
    super.key,
    required this.inputController,
    required this.resultController,
    required this.onPunctuate,
    required this.onClear,
    this.isProcessing = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        TextField(
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            hintText: 'Enter text without punctuation',
            labelText: 'Input',
            contentPadding:
                EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
          maxLines: 5,
          minLines: 3,
          controller: inputController,
          onTapOutside: (_) => FocusManager.instance.primaryFocus?.unfocus(),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            OutlinedButton(
              onPressed: isProcessing ? null : onPunctuate,
              child:
                  Text(isProcessing ? 'Processing...' : 'Punctuate'),
            ),
            const SizedBox(width: 8),
            OutlinedButton(
              onPressed: onClear,
              child: const Text('Clear'),
            ),
          ],
        ),
        const SizedBox(height: 8),
        TextField(
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            labelText: 'Result',
            contentPadding:
                EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
          maxLines: 5,
          minLines: 3,
          controller: resultController,
          readOnly: true,
        ),
      ],
    );
  }
}
