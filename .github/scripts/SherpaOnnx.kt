package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class EndpointRule(
    var mustContainNonSilence: Boolean,
    var minTrailingSilence: Float,
    var minUtteranceLength: Float,
)

data class EndpointConfig(
    var rule1: EndpointRule = EndpointRule(false, 2.4f, 0.0f),
    var rule2: EndpointRule = EndpointRule(true, 1.4f, 0.0f),
    var rule3: EndpointRule = EndpointRule(false, 0.0f, 20.0f)
)

data class OnlineTransducerModelConfig(
    var encoder: String,
    var decoder: String,
    var joiner: String,
    var numThreads: Int = 4,
		var debug: Boolean = false,
)

data class FeatureConfig(
    var sampleRate: Float = 16000.0f,
    var featureDim: Int = 80,
)

data class OnlineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OnlineTransducerModelConfig,
    var tokens: String,
    var endpointConfig: EndpointConfig = EndpointConfig(),
    var enableEndpoint: Boolean,
)

class SherpaOnnx(
    assetManager: AssetManager,
    var config: OnlineRecognizerConfig
) {
    private val ptr: Long

    init {
        ptr = new(assetManager, config)
    }

    protected fun finalize() {
        delete(ptr)
    }


    fun decodeSamples(samples: FloatArray) =
        decodeSamples(ptr, samples, sampleRate = config.featConfig.sampleRate)

    fun inputFinished() = inputFinished(ptr)
    fun reset() = reset(ptr)
    fun isEndpoint(): Boolean = isEndpoint(ptr)

    val text: String
        get() = getText(ptr)

    private external fun delete(ptr: Long)

    private external fun new(
        assetManager: AssetManager,
        config: OnlineRecognizerConfig,
    ): Long

    private external fun decodeSamples(ptr: Long, samples: FloatArray, sampleRate: Float)
    private external fun inputFinished(ptr: Long)
    private external fun getText(ptr: Long): String
    private external fun reset(ptr: Long)
    private external fun isEndpoint(ptr: Long): Boolean

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

fun getFeatureConfig(): FeatureConfig {
    val featConfig = FeatureConfig()
    featConfig.sampleRate = 16000.0f
    featConfig.featureDim = 80

    return featConfig
}
