// Copyright (c)  2024  Xiaomi Corporation
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
    var endpointConfig: EndpointConfig = EndpointConfig(),
    var enableEndpoint: Boolean = true,
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
    fun reset(recreate: Boolean = false) = reset(ptr, recreate = recreate)
    fun decode() = decode(ptr)
    fun isEndpoint(): Boolean = isEndpoint(ptr)
    fun isReady(): Boolean = isReady(ptr)

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
    private external fun reset(ptr: Long, recreate: Boolean)
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
    return FeatureConfig(sampleRate = sampleRate, featureDim = featureDim)
}

/*
Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own. (It should be straightforward to add a new model
by following the code)

@param type
0 - sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english

1 - csukuangfj/sherpa-onnx-lstm-zh-2023-02-20 (Chinese)

    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/lstm-transducer-models.html#csukuangfj-sherpa-onnx-lstm-zh-2023-02-20-chinese

2 - csukuangfj/sherpa-onnx-lstm-en-2023-02-17 (English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/lstm-transducer-models.html#csukuangfj-sherpa-onnx-lstm-en-2023-02-17-english

3,4 - pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
    https://huggingface.co/pkufool/icefall-asr-zipformer-streaming-wenetspeech-20230615
    3 - int8 encoder
    4 - float32 encoder

5 - csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en
    https://huggingface.co/csukuangfj/sherpa-onnx-streaming-paraformer-bilingual-zh-en

6 - sherpa-onnx-streaming-zipformer-en-2023-06-26
    https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26

7 - shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14 (French)
    https://huggingface.co/shaojieli/sherpa-onnx-streaming-zipformer-fr-2023-04-14

8 - csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
    encoder int8, decoder/joiner float32

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
            val modelDir = "icefall-asr-zipformer-streaming-wenetspeech-20230615"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/exp/encoder-epoch-12-avg-4-chunk-16-left-128.onnx",
                    decoder = "$modelDir/exp/decoder-epoch-12-avg-4-chunk-16-left-128.onnx",
                    joiner = "$modelDir/exp/joiner-epoch-12-avg-4-chunk-16-left-128.onnx",
                ),
                tokens = "$modelDir/data/lang_char/tokens.txt",
                modelType = "zipformer2",
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
