// Copyright (c)  2024  Xiaomi Corporation

class FeatureConfig {
  const FeatureConfig({this.sampleRate = 16000, this.featureDim = 80});

  @override
  String toString() {
    return 'FeatureConfig(sampleRate: $sampleRate, featureDim: $featureDim)';
  }

  final int sampleRate;
  final int featureDim;
}
