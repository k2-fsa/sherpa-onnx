package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class EndpointRule(
    var mustContainNonSilence: Boolean,
    var minTrailingSilence: Float,
    var minUtteranceLength: Float,
)

data class EndpointConfig(
    var rule1: EndpointRule = EndpointRule(false, 2.0f, 0.0f),
    var rule2: EndpointRule = EndpointRule(true, 1.2f, 0.0f),
    var rule3: EndpointRule = EndpointRule(false, 0.0f, 20.0f)
)

data class OnlineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = "",
)

data class OnlineParaformerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
)

data class OnlineZipformer2CtcModelConfig(
    var model: String = "",
)

data class OnlineModelConfig(
    var transducer: OnlineTransducerModelConfig = OnlineTransducerModelConfig(),
    var paraformer: OnlineParaformerModelConfig = OnlineParaformerModelConfig(),
    var zipformer2Ctc: OnlineZipformer2CtcModelConfig = OnlineZipformer2CtcModelConfig(),
    var tokens: String,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
    var modelType: String = "",
)

data class OnlineLMConfig(
    var model: String = "",
    var scale: Float = 0.5f,
)

data class FeatureConfig(
    var sampleRate: Int = 16000,
    var featureDim: Int = 80,
)

data class OnlineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OnlineModelConfig,
    var lmConfig: OnlineLMConfig = OnlineLMConfig(),
    var endpointConfig: EndpointConfig = EndpointConfig(),
    var enableEndpoint: Boolean = true,
    var decodingMethod: String = "greedy_search",
    var maxActivePaths: Int = 4,
    var hotwordsFile: String = "",
    var hotwordsScore: Float = 1.5f,
)

data class OfflineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = "",
)

data class OfflineParaformerModelConfig(
    var model: String = "",
)

data class OfflineWhisperModelConfig(
    var encoder: String = "",
    var decoder: String = "",
)

data class OfflineModelConfig(
    var transducer: OfflineTransducerModelConfig = OfflineTransducerModelConfig(),
    var paraformer: OfflineParaformerModelConfig = OfflineParaformerModelConfig(),
    var whisper: OfflineWhisperModelConfig = OfflineWhisperModelConfig(),
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
    var modelType: String = "",
    var tokens: String,
)

data class OfflineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OfflineModelConfig,
    // var lmConfig: OfflineLMConfig(), // TODO(fangjun): enable it
    var decodingMethod: String = "greedy_search",
    var maxActivePaths: Int = 4,
    var hotwordsFile: String = "",
    var hotwordsScore: Float = 1.5f,
)

class SherpaOnnx(
    assetManager: AssetManager? = null,
    var config: OnlineRecognizerConfig,
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

    val text: String
        get() = getText(ptr)

    private external fun delete(ptr: Long)

    private external fun new(
        assetManager: AssetManager,
        config: OnlineRecognizerConfig,
    ): Long

    private external fun newFromFile(
        config: OnlineRecognizerConfig,
    ): Long

    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Int)
    private external fun inputFinished(ptr: Long)
    private external fun getText(ptr: Long): String
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

class SherpaOnnxOffline(
    assetManager: AssetManager? = null,
    var config: OfflineRecognizerConfig,
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

    fun decode(samples: FloatArray, sampleRate: Int) = decode(ptr, samples, sampleRate)

    private external fun delete(ptr: Long)

    private external fun new(
        assetManager: AssetManager,
        config: OfflineRecognizerConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineRecognizerConfig,
    ): Long

    private external fun decode(ptr: Long, samples: FloatArray, sampleRate: Int): String

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
0 - csukuangfj/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23 (Chinese)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-zh-14m-2023-02-23
    encoder/joiner int8, decoder float32

1 - csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17 (English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html#csukuangfj-sherpa-onnx-streaming-zipformer-en-20m-2023-02-17-english
    encoder/joiner int8, decoder fp32

 */
fun getModelConfig(type: Int): OnlineModelConfig? {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        1 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
            return OnlineModelConfig(
                transducer = OnlineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }
    }
    return null
}

/*
Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own LM model. (It should be straightforward to train a new NN LM model
by following the code, https://github.com/k2-fsa/icefall/blob/master/icefall/rnn_lm/train.py)

@param type
0 - sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20 (Bilingual, Chinese + English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/zipformer-transducer-models.html#sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20-bilingual-chinese-english
 */
fun getOnlineLMConfig(type: Int): OnlineLMConfig {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
            return OnlineLMConfig(
                model = "$modelDir/with-state-epoch-99-avg-1.int8.onnx",
                scale = 0.5f,
            )
        }
    }
    return OnlineLMConfig()
}

// for English models, use a small value for rule2.minTrailingSilence, e.g., 0.8
fun getEndpointConfig(): EndpointConfig {
    return EndpointConfig(
        rule1 = EndpointRule(false, 2.4f, 0.0f),
        rule2 = EndpointRule(true, 0.8f, 0.0f),
        rule3 = EndpointRule(false, 0.0f, 20.0f)
    )
}

/*
Please see
https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
for a list of pre-trained models.

We only add a few here. Please change the following code
to add your own. (It should be straightforward to add a new model
by following the code)

@param type

0 - csukuangfj/sherpa-onnx-paraformer-zh-2023-03-28 (Chinese)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-paraformer/paraformer-models.html#csukuangfj-sherpa-onnx-paraformer-zh-2023-03-28-chinese
    int8

1 - icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04 (English)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#icefall-asr-multidataset-pruned-transducer-stateless7-2023-05-04-english
    encoder int8, decoder/joiner float32

2 - sherpa-onnx-whisper-tiny.en
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html#tiny-en
    encoder int8, decoder int8

3 - sherpa-onnx-whisper-base.en
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/whisper/tiny.en.html#tiny-en
    encoder int8, decoder int8

4 - pkufool/icefall-asr-zipformer-wenetspeech-20230615 (Chinese)
    https://k2-fsa.github.io/sherpa/onnx/pretrained_models/offline-transducer/zipformer-transducer-models.html#pkufool-icefall-asr-zipformer-wenetspeech-20230615-chinese
    encoder/joiner int8, decoder fp32

 */
fun getOfflineModelConfig(type: Int): OfflineModelConfig? {
    when (type) {
        0 -> {
            val modelDir = "sherpa-onnx-paraformer-zh-2023-03-28"
            return OfflineModelConfig(
                paraformer = OfflineParaformerModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "paraformer",
            )
        }

        1 -> {
            val modelDir = "icefall-asr-multidataset-pruned_transducer_stateless7-2023-05-04"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-30-avg-4.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-30-avg-4.onnx",
                    joiner = "$modelDir/joiner-epoch-30-avg-4.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        2 -> {
            val modelDir = "sherpa-onnx-whisper-tiny.en"
            return OfflineModelConfig(
                whisper = OfflineWhisperModelConfig(
                    encoder = "$modelDir/tiny.en-encoder.int8.onnx",
                    decoder = "$modelDir/tiny.en-decoder.int8.onnx",
                ),
                tokens = "$modelDir/tiny.en-tokens.txt",
                modelType = "whisper",
            )
        }

        3 -> {
            val modelDir = "sherpa-onnx-whisper-base.en"
            return OfflineModelConfig(
                whisper = OfflineWhisperModelConfig(
                    encoder = "$modelDir/base.en-encoder.int8.onnx",
                    decoder = "$modelDir/base.en-decoder.int8.onnx",
                ),
                tokens = "$modelDir/base.en-tokens.txt",
                modelType = "whisper",
            )
        }


        4 -> {
            val modelDir = "icefall-asr-zipformer-wenetspeech-20230615"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-12-avg-4.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-12-avg-4.onnx",
                    joiner = "$modelDir/joiner-epoch-12-avg-4.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer",
            )
        }

        5 -> {
            val modelDir = "sherpa-onnx-zipformer-multi-zh-hans-2023-9-2"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-20-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-20-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-20-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "zipformer2",
            )
        }

    }
    return null
}
