package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager


data class SpeakerEmbeddingExtractorConfig(
    val model: String,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)

class SpeakerEmbeddingExtractorStream(var ptr: Long) {
    fun acceptWaveform(samples: FloatArray, sampleRate: Int) = acceptWaveform(ptr, samples, sampleRate)

    fun inputFinished() = inputFinished(ptr)

    protected fun finalize() {
        delete(ptr)
        ptr = 0
    }

    private external fun myTest(ptr: Long, v: Array<FloatArray>)

    fun release() = finalize()
    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Int)

    private external fun inputFinished(ptr: Long)

    private external fun delete(ptr: Long)
    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

class SpeakerEmbeddingExtractor(
    assetManager: AssetManager? = null,
    config: SpeakerEmbeddingExtractorConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            new(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    protected fun finalize() {
        delete(ptr)
        ptr = 0
    }

    fun release() = finalize()

    fun createStream(): SpeakerEmbeddingExtractorStream {
        val p = createStream(ptr)
        return SpeakerEmbeddingExtractorStream(p)
    }

    fun isReady(stream: SpeakerEmbeddingExtractorStream) = isReady(ptr, stream.ptr)
    fun compute(stream: SpeakerEmbeddingExtractorStream) = compute(ptr, stream.ptr)

    private external fun new(
        assetManager: AssetManager,
        config: SpeakerEmbeddingExtractorConfig,
    ): Long

    private external fun newFromFile(
        config: SpeakerEmbeddingExtractorConfig,
    ): Long

    private external fun delete(ptr: Long)

    private external fun createStream(ptr: Long): Long

    private external fun isReady(ptr: Long, streamPtr: Long): Boolean


    private external fun compute(ptr: Long, streamPtr: Long): FloatArray

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
