// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Input/output controls for punctuation.
class PunctControls extends StatelessWidget {
  final TextEditingController inputController;
  final TextEditingController resultController;
  final VoidCallback onPunctuate;
  final VoidCallback onClear;
  final VoidCallback? onExample;
  final bool isProcessing;

  const PunctControls({
    super.key,
    required this.inputController,
    required this.resultController,
    required this.onPunctuate,
    required this.onClear,
    this.onExample,
    this.isProcessing = false,
  });

  static void _copyToClipboard(BuildContext context, String text) {
    if (text.isEmpty) return;
    Clipboard.setData(ClipboardData(text: text));
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Copied to clipboard'), duration: Duration(seconds: 1)),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        TextField(
          decoration: InputDecoration(
            border: const OutlineInputBorder(),
            hintText: 'Enter text without punctuation',
            labelText: 'Input',
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            suffixIcon: IconButton(
              icon: const Icon(Icons.copy, size: 18),
              tooltip: 'Copy input',
              onPressed: () => _copyToClipboard(context, inputController.text),
            ),
          ),
          style: const TextStyle(fontSize: 18),
          maxLines: 8,
          minLines: 5,
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
            if (onExample != null) ...[
              const SizedBox(width: 8),
              OutlinedButton(
                onPressed: onExample,
                child: const Text('Example'),
              ),
            ],
          ],
        ),
        const SizedBox(height: 8),
        TextField(
          decoration: InputDecoration(
            border: const OutlineInputBorder(),
            labelText: 'Result',
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            suffixIcon: IconButton(
              icon: const Icon(Icons.copy, size: 18),
              tooltip: 'Copy result',
              onPressed: () => _copyToClipboard(context, resultController.text),
            ),
          ),
          style: const TextStyle(fontSize: 18),
          maxLines: 8,
          minLines: 5,
          controller: resultController,
          readOnly: true,
        ),
      ],
    );
  }
}
