// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Input controls for TTS: speaker ID, speed slider, text input, action buttons.
class TtsControls extends StatelessWidget {
  final int maxSpeakerID;
  final double speed;
  final ValueChanged<double> onSpeedChanged;
  final TextEditingController textController;
  final TextEditingController sidController;
  final VoidCallback onGenerate;
  final VoidCallback onClear;
  final VoidCallback? onStop;
  final bool isGenerating;

  const TtsControls({
    super.key,
    required this.maxSpeakerID,
    required this.speed,
    required this.onSpeedChanged,
    required this.textController,
    required this.sidController,
    required this.onGenerate,
    required this.onClear,
    this.onStop,
    this.isGenerating = false,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        TextField(
          decoration: InputDecoration(
            labelText: 'Speaker ID (0-$maxSpeakerID)',
            hintText: 'Speaker ID',
            isDense: true,
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
          keyboardType: TextInputType.number,
          maxLines: 1,
          controller: sidController,
          onTapOutside: (_) => FocusManager.instance.primaryFocus?.unfocus(),
          inputFormatters: [FilteringTextInputFormatter.digitsOnly],
        ),
        Slider(
          label: 'Speed ${speed.toStringAsPrecision(2)}',
          min: 0.5,
          max: 3.0,
          divisions: 25,
          value: speed,
          onChanged: onSpeedChanged,
        ),
        Expanded(
          child: TextField(
            decoration: const InputDecoration(
              border: OutlineInputBorder(),
              hintText: 'Enter text to synthesize',
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            ),
            expands: true,
            maxLines: null,
            minLines: null,
            textAlignVertical: TextAlignVertical.top,
            controller: textController,
            onTapOutside: (_) => FocusManager.instance.primaryFocus?.unfocus(),
          ),
        ),
        const SizedBox(height: 4),
        Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            OutlinedButton(
              onPressed: isGenerating ? null : onGenerate,
              child: Text(isGenerating ? 'Generating...' : 'Generate'),
            ),
            const SizedBox(width: 5),
            OutlinedButton(
              onPressed: onClear,
              child: const Text('Clear'),
            ),
            if (onStop != null) ...[
              const SizedBox(width: 5),
              OutlinedButton(
                onPressed: onStop,
                child: const Text('Stop'),
              ),
            ],
          ],
        ),
      ],
    );
  }
}
