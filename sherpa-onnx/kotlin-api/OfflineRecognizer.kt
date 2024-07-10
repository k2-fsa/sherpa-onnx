package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OfflineRecognizerResult(
    val text: String,
    val tokens: Array<String>,
    val timestamps: FloatArray,
)

data class OfflineTransducerModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var joiner: String = "",
)

data class OfflineParaformerModelConfig(
    var model: String = "",
)

data class OfflineNemoEncDecCtcModelConfig(
    var model: String = "",
)

data class OfflineWhisperModelConfig(
    var encoder: String = "",
    var decoder: String = "",
    var language: String = "en", // Used with multilingual model
    var task: String = "transcribe", // transcribe or translate
    var tailPaddings: Int = 1000, // Padding added at the end of the samples
)

data class OfflineModelConfig(
    var transducer: OfflineTransducerModelConfig = OfflineTransducerModelConfig(),
    var paraformer: OfflineParaformerModelConfig = OfflineParaformerModelConfig(),
    var whisper: OfflineWhisperModelConfig = OfflineWhisperModelConfig(),
    var nemo: OfflineNemoEncDecCtcModelConfig = OfflineNemoEncDecCtcModelConfig(),
    var teleSpeech: String = "",
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
    var modelType: String = "",
    var tokens: String,
    var modelingUnit: String = "",
    var bpeVocab: String = "",
)

data class OfflineRecognizerConfig(
    var featConfig: FeatureConfig = FeatureConfig(),
    var modelConfig: OfflineModelConfig,
    // var lmConfig: OfflineLMConfig(), // TODO(fangjun): enable it
    var decodingMethod: String = "greedy_search",
    var maxActivePaths: Int = 4,
    var hotwordsFile: String = "",
    var hotwordsScore: Float = 1.5f,
    var ruleFsts: String = "",
    var ruleFars: String = "",
)

class OfflineRecognizer(
    assetManager: AssetManager? = null,
    config: OfflineRecognizerConfig,
) {
    private val ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    protected fun finalize() {
        delete(ptr)
    }

    fun release() = finalize()

    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        return OfflineStream(p)
    }

    fun getResult(stream: OfflineStream): OfflineRecognizerResult {
        val objArray = getResult(stream.ptr)

        val text = objArray[0] as String
        val tokens = objArray[1] as Array<String>
        val timestamps = objArray[2] as FloatArray
        return OfflineRecognizerResult(text = text, tokens = tokens, timestamps = timestamps)
    }

    fun decode(stream: OfflineStream) = decode(ptr, stream.ptr)

    private external fun delete(ptr: Long)

    private external fun createStream(ptr: Long): Long

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OfflineRecognizerConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineRecognizerConfig,
    ): Long

    private external fun decode(ptr: Long, streamPtr: Long)

    private external fun getResult(streamPtr: Long): Array<Any>

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
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
                modelType = "transducer",
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
                modelType = "transducer",
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
                modelType = "transducer",
            )
        }

        6 -> {
            val modelDir = "sherpa-onnx-nemo-ctc-en-citrinet-512"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        7 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-be-de-en-es-fr-hr-it-pl-ru-uk-20k"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        8 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-en-24500"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        9 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-en-de-es-fr-14288"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        10 -> {
            val modelDir = "sherpa-onnx-nemo-fast-conformer-ctc-es-1424"
            return OfflineModelConfig(
                nemo = OfflineNemoEncDecCtcModelConfig(
                    model = "$modelDir/model.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
            )
        }

        11 -> {
            val modelDir = "sherpa-onnx-telespeech-ctc-int8-zh-2024-06-04"
            return OfflineModelConfig(
                teleSpeech = "$modelDir/model.int8.onnx",
                tokens = "$modelDir/tokens.txt",
                modelType = "telespeech_ctc",
            )
        }

        12 -> {
            val modelDir = "sherpa-onnx-zipformer-thai-2024-06-20"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-12-avg-5.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-12-avg-5.onnx",
                    joiner = "$modelDir/joiner-epoch-12-avg-5.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }

        13 -> {
            val modelDir = "sherpa-onnx-zipformer-korean-2024-06-24"
            return OfflineModelConfig(
                transducer = OfflineTransducerModelConfig(
                    encoder = "$modelDir/encoder-epoch-99-avg-1.int8.onnx",
                    decoder = "$modelDir/decoder-epoch-99-avg-1.onnx",
                    joiner = "$modelDir/joiner-epoch-99-avg-1.int8.onnx",
                ),
                tokens = "$modelDir/tokens.txt",
                modelType = "transducer",
            )
        }
    }
    return null
}
