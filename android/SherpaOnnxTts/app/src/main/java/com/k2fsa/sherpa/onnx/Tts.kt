// Copyright (c)  2023  Xiaomi Corporation
package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OfflineTtsVitsModelConfig(
    var model: String,
    var lexicon: String,
    var tokens: String,
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
        if (assetManager != null) {
            ptr = new(assetManager, config)
        } else {
            ptr = newFromFile(config)
        }
    }

    fun generate(
        text: String,
        sid: Int = 0,
        speed: Float = 1.0f
    ): GeneratedAudio {
        var objArray = generateImpl(ptr, text = text, sid = sid, speed = speed)
        return GeneratedAudio(
            samples = objArray[0] as FloatArray,
            sampleRate = objArray[1] as Int
        )
    }

    fun allocate(assetManager: AssetManager? = null) {
        if (ptr == 0L) {
            if (assetManager != null) {
                ptr = new(assetManager, config)
            } else {
                ptr = newFromFile(config)
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
        delete(ptr)
    }

    private external fun new(
        assetManager: AssetManager,
        config: OfflineTtsConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineTtsConfig,
    ): Long

    private external fun delete(ptr: Long)

    // The returned array has two entries:
    //  - the first entry is an 1-D float array containing audio samples.
    //    Each sample is normalized to the range [-1, 1]
    //  - the second entry is the sample rate
    external fun generateImpl(
        ptr: Long,
        text: String,
        sid: Int = 0,
        speed: Float = 1.0f
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
//
// You can change the type as you wish
fun getOfflineTtsConfig(type: Int, debug: Boolean = false): OfflineTtsConfig? {
    when (type) {
        0 -> {
            val modelDir = "vits-vctk"
            return OfflineTtsConfig(
                model = OfflineTtsModelConfig(
                    vits = OfflineTtsVitsModelConfig(
                        model = "$modelDir/vits-vctk.onnx",
                        lexicon = "$modelDir/lexicon.txt",
                        tokens = "$modelDir/tokens.txt"
                    ),
                    numThreads = 2,
                    debug = debug,
                    provider = "cpu",
                )
            )
        }

        1 -> {
            val modelDir = "vits-zh-aishell3"
            return OfflineTtsConfig(
                model = OfflineTtsModelConfig(
                    vits = OfflineTtsVitsModelConfig(
                        model = "$modelDir/vits-aishell3.onnx",
                        lexicon = "$modelDir/lexicon.txt",
                        tokens = "$modelDir/tokens.txt"
                    ),
                    numThreads = 2,
                    debug = debug,
                    provider = "cpu",
                )
            )
        }
    }

    println("Unsupported type $type")

    return null

}
