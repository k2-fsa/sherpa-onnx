package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager
import android.util.Log

private val TAG = "sherpa-onnx"

data class SpokenLanguageIdentificationWhisperConfig (
    var encoder: String,
    var decoder: String,
    var tailPaddings: Int = -1,
)

data class SpokenLanguageIdentificationConfig (
    var whisper: SpokenLanguageIdentificationWhisperConfig,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)

class SpokenLanguageIdentification (
    assetManager: AssetManager? = null,
    config: SpokenLanguageIdentificationConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        return OfflineStream(p)
    }

    fun compute(stream: OfflineStream) =  compute(ptr, stream.ptr)

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: SpokenLanguageIdentificationConfig,
    ): Long

    private external fun newFromFile(
        config: SpokenLanguageIdentificationConfig,
    ): Long

    private external fun delete(ptr: Long)

    private external fun createStream(ptr: Long): Long

    private external fun compute(ptr: Long, streamPtr: Long): String

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
