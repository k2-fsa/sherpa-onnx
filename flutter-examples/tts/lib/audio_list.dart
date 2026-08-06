// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:audioplayers/audioplayers.dart';

import './generated_audio.dart';
import './web_audio.dart' if (dart.library.io) './web_audio_stub.dart'
    as web_audio;

/// Displays a list of generated audio items with play, stop, download, and save.
class AudioList extends StatelessWidget {
  final List<GeneratedAudioItem> items;
  final AudioPlayer? player;
  final void Function(GeneratedAudioItem item, int index) onSaveAs;

  const AudioList({
    super.key,
    required this.items,
    required this.player,
    required this.onSaveAs,
  });

  @override
  Widget build(BuildContext context) {
    if (items.isEmpty) return const SizedBox.shrink();

    return ListView.builder(
      itemCount: items.length,
      itemBuilder: (context, index) {
        final item = items[index];
        final idx = items.length - index;
        return Card(
          margin: const EdgeInsets.symmetric(vertical: 4),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
            child: Row(
              children: [
                // Play button.
                IconButton(
                  icon: const Icon(Icons.play_arrow),
                  tooltip: 'Play',
                  onPressed: () => _play(item),
                ),
                // Stop button.
                IconButton(
                  icon: const Icon(Icons.stop),
                  tooltip: 'Stop',
                  onPressed: () => _stop(),
                ),
                // Label and duration.
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        '$idx. ${item.label}',
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                      Text(
                        '${item.duration.toStringAsPrecision(3)}s | '
                        'RTF ${item.duration > 0 ? (item.elapsed / item.duration).toStringAsPrecision(3) : '-'}',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ],
                  ),
                ),
                // Download (web only).
                if (kIsWeb)
                  IconButton(
                    icon: const Icon(Icons.download),
                    tooltip: 'Download',
                    onPressed: () {
                      web_audio.downloadWavBytes(
                          item.wavBytes!, '$idx-${item.label}.wav');
                    },
                  ),
                // Save as.
                IconButton(
                  icon: const Icon(Icons.save_as),
                  tooltip: 'Save as',
                  onPressed: () => onSaveAs(item, idx),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Future<void> _play(GeneratedAudioItem item) async {
    if (kIsWeb) {
      web_audio.playWavBytes(item.wavBytes!);
    } else {
      await player?.stop();
      await player?.play(DeviceFileSource(item.filePath!));
    }
  }

  Future<void> _stop() async {
    if (kIsWeb) {
      web_audio.stopPlayback();
    } else {
      await player?.stop();
    }
  }
}
