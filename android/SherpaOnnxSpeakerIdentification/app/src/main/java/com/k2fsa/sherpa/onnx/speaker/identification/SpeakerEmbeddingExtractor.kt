package com.k2fsa.sherpa.onnx.speaker.identification

import android.content.res.AssetManager

data class SpeakerEmbeddingExtractorConfig(
    val model: String,
)

class SpeakerEmbeddingExtractor(
    assetManager: AssetManager? = null,
    config: SpeakerEmbeddingExtractorConfig,
) {
    private val ptr: Long

    init {
        ptr = if (assetManager != null) {
            new(assetManager, config)
        } else {
            newFromFile(config)
        }
    }

    protected fun finalize() {
        delete(ptr)
    }

    private external fun new(
        assetManager: AssetManager,
        config: SpeakerEmbeddingExtractorConfig,
    ): Long

    private external fun newFromFile(
        config: SpeakerEmbeddingExtractorConfig,
    ): Long

    private external fun delete(ptr: Long)

    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Int)

    private external fun inputFinished(ptr: Long)

    private external fun isReady(ptr: Long): Boolean

    private external fun compute(ptr: Long): FloatArray

    private external fun reset(ptr: Long)

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}