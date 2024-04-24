// Copyright (c)  2023  Xiaomi Corporation
package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OfflineTtsVitsModelConfig(
    var model: String,
    var lexicon: String = "",
    var tokens: String,
    var dataDir: String = "",
    var dictDir: String = "",
    var noiseScale: Float = 0.667f,
    var noiseScaleW: Float = 0.8f,
    var lengthScale: Float = 1.0f,
)

data class OfflineTtsModelConfig(
    var vits: OfflineTtsVitsModelConfig,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)

data class OfflineTtsConfig(
    var model: OfflineTtsModelConfig,
    var ruleFsts: String = "",
    var ruleFars: String = "",
    var maxNumSentences: Int = 1,
)

class GeneratedAudio(
    val samples: FloatArray,
    val sampleRate: Int,
) {
    fun save(filename: String) =
        saveImpl(filename = filename, samples = samples, sampleRate = sampleRate)

    private external fun saveImpl(
        filename: String,
        samples: FloatArray,
        sampleRate: Int
    ): Boolean
}

class OfflineTts(
    assetManager: AssetManager? = null,
    var config: OfflineTtsConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    fun sampleRate() = getSampleRate(ptr)

    fun numSpeakers() = getNumSpeakers(ptr)

    fun generate(
        text: String,
        sid: Int = 0,
        speed: Float = 1.0f
    ): GeneratedAudio {
        val objArray = generateImpl(ptr, text = text, sid = sid, speed = speed)
        return GeneratedAudio(
            samples = objArray[0] as FloatArray,
            sampleRate = objArray[1] as Int
        )
    }

    fun generateWithCallback(
        text: String,
        sid: Int = 0,
        speed: Float = 1.0f,
        callback: (samples: FloatArray) -> Unit
    ): GeneratedAudio {
        val objArray = generateWithCallbackImpl(
            ptr,
            text = text,
            sid = sid,
            speed = speed,
            callback = callback
        )
        return GeneratedAudio(
            samples = objArray[0] as FloatArray,
            sampleRate = objArray[1] as Int
        )
    }

    fun allocate(assetManager: AssetManager? = null) {
        if (ptr == 0L) {
            ptr = if (assetManager != null) {
                newFromAsset(assetManager, config)
            } else {
                newFromFile(config)
            }
        }
    }

    fun free() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OfflineTtsConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineTtsConfig,
    ): Long

    private external fun delete(ptr: Long)
    private external fun getSampleRate(ptr: Long): Int
    private external fun getNumSpeakers(ptr: Long): Int

    // The returned array has two entries:
    //  - the first entry is an 1-D float array containing audio samples.
    //    Each sample is normalized to the range [-1, 1]
    //  - the second entry is the sample rate
    private external fun generateImpl(
        ptr: Long,
        text: String,
        sid: Int = 0,
        speed: Float = 1.0f
    ): Array<Any>

    private external fun generateWithCallbackImpl(
        ptr: Long,
        text: String,
        sid: Int = 0,
        speed: Float = 1.0f,
        callback: (samples: FloatArray) -> Unit
    ): Array<Any>

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

// please refer to
// https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/index.html
// to download models
fun getOfflineTtsConfig(
    modelDir: String,
    modelName: String,
    lexicon: String,
    dataDir: String,
    dictDir: String,
    ruleFsts: String,
    ruleFars: String
): OfflineTtsConfig {
    return OfflineTtsConfig(
        model = OfflineTtsModelConfig(
            vits = OfflineTtsVitsModelConfig(
                model = "$modelDir/$modelName",
                lexicon = "$modelDir/$lexicon",
                tokens = "$modelDir/tokens.txt",
                dataDir = dataDir,
                dictDir = dictDir,
            ),
            numThreads = 2,
            debug = true,
            provider = "cpu",
        ),
        ruleFsts = ruleFsts,
        ruleFars = ruleFars,
    )
}
