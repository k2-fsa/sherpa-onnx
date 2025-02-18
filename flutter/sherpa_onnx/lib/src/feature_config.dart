// Copyright (c)  2024  Xiaomi Corporation

class FeatureConfig {
  const FeatureConfig({this.sampleRate = 16000, this.featureDim = 80});

  factory FeatureConfig.fromJson(Map<String, dynamic> json) {
    return FeatureConfig(
      sampleRate: json['sampleRate'] as int? ?? 16000,
      featureDim: json['featureDim'] as int? ?? 80,
    );
  }

  @override
  String toString() {
    return 'FeatureConfig(sampleRate: $sampleRate, featureDim: $featureDim)';
  }

  Map<String, dynamic> toJson() => {
        'sampleRate': sampleRate,
        'featureDim': featureDim,
      };

  final int sampleRate;
  final int featureDim;
}
