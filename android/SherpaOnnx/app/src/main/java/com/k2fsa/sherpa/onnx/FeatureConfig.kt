package com.k2fsa.sherpa.onnx

data class FeatureConfig(
    var sampleRate: Int = 16000,
    var featureDim: Int = 80,
)

fun getFeatureConfig(
    sampleRate: Int = 16000,
    featureDim: Int = 80,
): FeatureConfig {
    return FeatureConfig(
        sampleRate = sampleRate,
        featureDim = featureDim,
    )
}
