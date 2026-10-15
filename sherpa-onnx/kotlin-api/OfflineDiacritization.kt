package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OfflineDiacritizationModelConfig(
    var cattEncoder: String = "",
    var cattDecoder: String = "",
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)


data class OfflineDiacritizationConfig(
    var model: OfflineDiacritizationModelConfig,
)

class OfflineDiacritization(
    assetManager: AssetManager? = null,
    config: OfflineDiacritizationConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
        require(ptr != 0L) {
            "Invalid OfflineDiacritizationConfig: failed to create native OfflineDiacritization"
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    fun addDiacritics(text: String) = addDiacritics(ptr, text)

    private external fun delete(ptr: Long)

    private external fun addDiacritics(ptr: Long, text: String): String

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OfflineDiacritizationConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineDiacritizationConfig,
    ): Long

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
