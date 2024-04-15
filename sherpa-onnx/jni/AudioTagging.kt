package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager
import android.util.Log

private val TAG = "sherpa-onnx"

data class OfflineZipformerAudioTaggingModelConfig (
    val model: String,
)

data class AudioTaggingModelConfig (
    var zipformer: OfflineZipformerAudioTaggingModelConfig,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)

data class AudioTaggingConfig (
    var model: AudioTaggingModelConfig,
    var labels: String,
    var topK: Int = 5,
)

data class AudioEvent (
    val name: String,
    val index: Int,
    val prob: Float,
)

class AudioTagging(
    assetManager: AssetManager? = null,
    config: AudioTaggingConfig,
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
      if(ptr != 0) {
        delete(ptr)
        ptr = 0
      }
    }

    fun release() = finalize()

    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        return OfflineStream(p)
    }

    // fun compute(stream: OfflineStream, topK: Int=-1): Array<AudioEvent> {
    fun compute(stream: OfflineStream, topK: Int=-1): Array<Any> {
      var events :Array<Any> = compute(ptr, stream.ptr, topK)
    }

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: AudioTaggingConfig,
    ): Long

    private external fun newFromFile(
        config: AudioTaggingConfig,
    ): Long

    private external fun delete(ptr: Long)

    private external fun createStream(ptr: Long): Long

    private external fun compute(ptr: Long, streamPtr: Long, topK: Int): Array<Any>

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
