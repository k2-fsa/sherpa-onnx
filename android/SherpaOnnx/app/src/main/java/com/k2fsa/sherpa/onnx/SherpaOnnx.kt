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
    var tokens: String,
    var numThreads: Int = 2,
    var debug: Boolean = false,
)

data class FeatureConfig(
    var sampleRate: Int = 16000,
    var featureDim: Int = 80,
)

data class OnlineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OnlineTransducerModelConfig,
    var endpointConfig: EndpointConfig = EndpointConfig(),
    var enableEndpoint: Boolean = true,
    var decodingMethod: String = "greedy_search",
    var maxActivePaths: Int = 4,
)

class SherpaOnnx(
    assetManager: AssetManager, var config: OnlineRecognizerConfig
) {
    private val ptr: Long

    init {
        ptr = new(assetManager, config)
    }

    protected fun finalize() {
        delete(ptr)
    }


    fun acceptWaveform(samples: FloatArray, sampleRate: Int) =
        acceptWaveform(ptr, samples, sampleRate)

    fun inputFinished() = inputFinished(ptr)
    fun reset() = reset(ptr)
    fun decode() = decode(ptr)
    fun isEndpoint(): Boolean = isEndpoint(ptr)
    fun isReady(): Boolean = isReady(ptr)

    val text: String
        get() = getText(ptr)

    private external fun delete(ptr: Long)

    private external fun new(
        assetManager: AssetManager,
        config: OnlineRecognizerConfig,
    ): Long

    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Int)
    private external fun inputFinished(ptr: Long)
    private external fun getText(ptr: Long): String
    private external fun reset(ptr: Long)
    private external fun decode(ptr: Long)
    private external fun isEndpoint(ptr: Long): Boolean
    private external fun isReady(ptr: Long): Boolean

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

fun getFeatureConfig(sampleRate: Int, featureDim: Int): FeatureConfig {
    return FeatureConfig(sampleRate=sampleRate, featureDim=featureDim)
}

/*
Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own. (It should be straightforward to add a new model
by following the code)

@param type
0 - sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

1 - csukuangfj/sherpa-onnx-lstm-zh-2023-02-20 (Chinese)

    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/lstm-transducer-models.html#csukuangfj-sherpa-onnx-lstm-zh-2023-02-20-chinese

2 - csukuangfj/sherpa-onnx-lstm-en-2023-02-17 (English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/lstm-transducer-models.html#csukuangfj-sherpa-onnx-lstm-en-2023-02-17-english
 */
fun getModelConfig(type: Int): OnlineTransducerModelConfig? {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
            return OnlineTransducerModelConfig(
                encoder = "$modelDir/encoder-epoch-99-avg-1.onnx",
                decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                joiner = "$modelDir/joiner-epoch-99-avg-1.onnx",
                tokens = "$modelDir/tokens.txt",
            )
        }
        1 -> {
            val modelDir = "sherpa-onnx-lstm-zh-2023-02-20"
            return OnlineTransducerModelConfig(
                encoder = "$modelDir/encoder-epoch-11-avg-1.onnx",
                decoder = "$modelDir/decoder-epoch-11-avg-1.onnx",
                joiner = "$modelDir/joiner-epoch-11-avg-1.onnx",
                tokens = "$modelDir/tokens.txt",
            )
        }

        2 -> {
            val modelDir = "sherpa-onnx-lstm-en-2023-02-17"
            return OnlineTransducerModelConfig(
                encoder = "$modelDir/encoder-epoch-99-avg-1.onnx",
                decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                joiner = "$modelDir/joiner-epoch-99-avg-1.onnx",
                tokens = "$modelDir/tokens.txt",
            )
        }
    }
    return null;
}

fun getEndpointConfig(): EndpointConfig {
    return EndpointConfig(
        rule1 = EndpointRule(false, 2.4f, 0.0f),
        rule2 = EndpointRule(true, 1.4f, 0.0f),
        rule3 = EndpointRule(false, 0.0f, 20.0f)
    )
}
