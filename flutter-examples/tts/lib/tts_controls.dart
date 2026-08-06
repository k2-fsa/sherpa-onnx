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

  // Reference audio support (shown only for Pocket TTS).
  final bool showReferenceAudio;
  final String? referenceAudioLabel;
  final VoidCallback? onPickReferenceAudio;
  final VoidCallback? onPlayReferenceAudio;
  final VoidCallback? onStopReferenceAudio;
  final bool isRefPlaying;
  final TextEditingController? numStepsController;

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
    this.showReferenceAudio = false,
    this.referenceAudioLabel,
    this.onPickReferenceAudio,
    this.onPlayReferenceAudio,
    this.onStopReferenceAudio,
    this.isRefPlaying = false,
    this.numStepsController,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        if (showReferenceAudio) ...[
          Row(
            children: [
              Expanded(
                child: Text(
                  referenceAudioLabel != null
                      ? 'Reference: $referenceAudioLabel'
                      : 'No reference audio selected',
                  style: TextStyle(
                    fontSize: 13,
                    color: referenceAudioLabel != null ? null : Colors.grey,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
              ),
              const SizedBox(width: 8),
              OutlinedButton(
                onPressed: onPickReferenceAudio,
                child: const Text('Pick WAV'),
              ),
              if (referenceAudioLabel != null) ...[
                const SizedBox(width: 5),
                OutlinedButton(
                  onPressed: isRefPlaying ? onStopReferenceAudio : onPlayReferenceAudio,
                  child: Text(isRefPlaying ? 'Stop' : 'Play'),
                ),
              ],
            ],
          ),
          const SizedBox(height: 4),
          TextField(
            decoration: const InputDecoration(
              labelText: 'Num steps',
              hintText: '5',
              isDense: true,
              contentPadding:
                  EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            ),
            keyboardType: TextInputType.number,
            maxLines: 1,
            controller: numStepsController,
            onTapOutside: (_) => FocusManager.instance.primaryFocus?.unfocus(),
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
          ),
          const SizedBox(height: 4),
        ],
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
        TextField(
          decoration: const InputDecoration(
            border: OutlineInputBorder(),
            hintText: 'Enter text to synthesize',
            contentPadding:
                EdgeInsets.symmetric(horizontal: 12, vertical: 8),
          ),
          maxLines: 8,
          minLines: 4,
          controller: textController,
          onTapOutside: (_) => FocusManager.instance.primaryFocus?.unfocus(),
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
