// Copyright (c)  2026  Xiaomi Corporation
import 'dart:io';

import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;

void main() {
  sherpa_onnx.initBindings();

  final modelDir = './english-mfa-v3.1.0';
  final wav = './english-mfa-v3.1.0/en.wav';
  // The transcript must match the wav.
  final text =
      'ask not what your country can do for you, ask what you can do for your country.';

  final config = sherpa_onnx.ForcedAlignerConfig(
    modelDir: modelDir,
    lexicon: '$modelDir/lexicon.txt',
  );

  final aligner = sherpa_onnx.ForcedAligner(config: config);
  final stream = aligner.createStream();

  final wave = sherpa_onnx.readWave(wav);
  stream.acceptWaveform(samples: wave.samples, sampleRate: wave.sampleRate);

  final start = DateTime.now();
  final result = aligner.compute(stream: stream, text: text);
  final elapsed =
      DateTime.now().difference(start).inMicroseconds / 1000000.0;

  if (result == null) {
    print('Alignment failed');
    exit(1);
  }

  // tiers[0] is the "phones" tier; every interval has a phone symbol.
  print('phones:');
  for (final iv in result.tiers[0].intervals) {
    print('${iv.start.toStringAsFixed(3)} - ${iv.end.toStringAsFixed(3)}'
        '  ${iv.text}');
  }

  // tiers[1] is the "words" tier; an empty text marks a silence gap.
  print('words:');
  for (final iv in result.tiers[1].intervals) {
    final label = iv.text.isEmpty ? '<silence>' : iv.text;
    print('${iv.start.toStringAsFixed(3)} - ${iv.end.toStringAsFixed(3)}'
        '  $label');
  }

  const output = 'en.TextGrid';
  File(output).writeAsStringSync(result.textGrid);
  print('Saved to $output');

  final duration = wave.samples.length / wave.sampleRate;
  final rtf = elapsed / duration;
  print('Elapsed seconds: ${elapsed.toStringAsFixed(3)}');
  print('Audio duration in seconds: ${duration.toStringAsFixed(3)}');
  print('RTF: ${elapsed.toStringAsFixed(3)}/${duration.toStringAsFixed(3)}'
      ' = ${rtf.toStringAsFixed(3)}');

  stream.free();
  aligner.free();
}
