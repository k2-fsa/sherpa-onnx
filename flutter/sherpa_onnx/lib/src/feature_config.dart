// Copyright (c)  2024  Xiaomi Corporation

/// Feature extraction settings shared by recognizers and keyword spotting.
///
/// In most cases the defaults of 16 kHz audio and 80-dimensional filterbank
/// features should match the model packages provided in the repository.
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
