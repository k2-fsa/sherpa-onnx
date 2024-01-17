// Copyright (c)  2024  Xiaomi Corporation
package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OnlineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = "",
)

data class OnlineModelConfig(
    var transducer: OnlineTransducerModelConfig = OnlineTransducerModelConfig(),
    var tokens: String,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
    var modelType: String = "",
)

data class FeatureConfig(
    var sampleRate: Int = 16000,
    var featureDim: Int = 80,
)

data class KeywordSpotterConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OnlineModelConfig,
    var maxActivePaths: Int = 4,
    var keywordsFile: String = "keywords.txt",
    var keywordsScore: Float = 1.5f,
    var keywordsThreshold: Float = 0.25f,
    var numTrailingBlanks: Int = 2,
)

class SherpaOnnxKws(
    assetManager: AssetManager? = null,
    var config: KeywordSpotterConfig,
) {
    private val ptr: Long

    init {
        if (assetManager != null) {
            ptr = new(assetManager, config)
        } else {
            ptr = newFromFile(config)
        }
    }

    protected fun finalize() {
        delete(ptr)
    }

    fun acceptWaveform(samples: FloatArray, sampleRate: Int) =
        acceptWaveform(ptr, samples, sampleRate)

    fun inputFinished() = inputFinished(ptr)
    fun decode() = decode(ptr)
    fun isReady(): Boolean = isReady(ptr)
    fun reset(keywords: String): Boolean = reset(ptr, keywords)

    val keyword: String
        get() = getKeyword(ptr)

    private external fun delete(ptr: Long)

    private external fun new(
        assetManager: AssetManager,
        config: KeywordSpotterConfig,
    ): Long

    private external fun newFromFile(
        config: KeywordSpotterConfig,
    ): Long

    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Int)
    private external fun inputFinished(ptr: Long)
    private external fun getKeyword(ptr: Long): String
    private external fun reset(ptr: Long, keywords: String): Boolean
    private external fun decode(ptr: Long)
    private external fun isReady(ptr: Long): Boolean

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

fun getFeatureConfig(sampleRate: Int, featureDim: Int): FeatureConfig {
    return FeatureConfig(sampleRate = sampleRate, featureDim = featureDim)
}

/*
Please see
https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own. (It should be straightforward to add a new model
by following the code)

@param type
0 - sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01 (Chinese)
    https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/summary

1 - sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01 (English)
    https://www.modelscope.cn/models/pkufool/sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01/summary

 */
fun getModelConfig(type: Int): OnlineModelConfig? {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                    decoder = "$modelDir/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                    joiner = "$modelDir/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer2",
            )
        }

        1 -> {
            val modelDir = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                    decoder = "$modelDir/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
                    joiner = "$modelDir/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer2",
            )
        }

    }
    return null;
}

/*
 * Get the default keywords for each model.
 * Caution: The types and modelDir should be the same as those in getModelConfig
 * function above.
 */ 
fun getKeywordsFile(type: Int) : String {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
            return "$modelDir/keywords.txt"
        }

        1 -> {
            val modelDir = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
            return "$modelDir/keywords.txt"
        }

    }
    return "";
}
