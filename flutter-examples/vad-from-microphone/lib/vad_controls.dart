// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

/// Settings panel for VAD parameters.
class VadControls extends StatelessWidget {
  final TextEditingController thresholdController;
  final TextEditingController minSilenceController;
  final TextEditingController minSpeechController;
  final TextEditingController maxSpeechController;

  const VadControls({
    super.key,
    required this.thresholdController,
    required this.minSilenceController,
    required this.minSpeechController,
    required this.maxSpeechController,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      child: ExpansionTile(
        title: const Text('VAD Settings'),
        initiallyExpanded: false,
        children: [
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            child: Column(
              children: [
                _buildField('Threshold (0.0–1.0)', thresholdController, '0.8'),
                _buildField(
                    'Min Silence Duration (s)', minSilenceController, '0.2'),
                _buildField(
                    'Min Speech Duration (s)', minSpeechController, '0.2'),
                _buildField(
                    'Max Speech Duration (s)', maxSpeechController, '12.0'),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildField(
      String label, TextEditingController controller, String hint) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: TextField(
        decoration: InputDecoration(
          labelText: label,
          hintText: hint,
          isDense: true,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        ),
        keyboardType: const TextInputType.numberWithOptions(decimal: true),
        controller: controller,
      ),
    );
  }
}
