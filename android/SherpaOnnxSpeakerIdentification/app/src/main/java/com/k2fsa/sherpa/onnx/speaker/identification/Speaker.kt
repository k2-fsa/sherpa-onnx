package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager
import android.util.Log

private val TAG = "sherpa-onnx"
data class SpeakerEmbeddingExtractorConfig(
    val model: String,
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)

class SpeakerEmbeddingExtractorStream(var ptr: Long) {
    fun acceptWaveform(samples: FloatArray, sampleRate: Int) =
        acceptWaveform(ptr, samples, sampleRate)

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
    fun dim() = dim(ptr)

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

    private external fun dim(ptr: Long): Int

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

class SpeakerEmbeddingManager(val dim: Int) {
    private var ptr: Long

    init {
        ptr = new(dim)
    }

    protected fun finalize() {
        delete(ptr)
        ptr = 0
    }

    fun release() = finalize()
    fun add(name: String, embedding: FloatArray) = add(ptr, name, embedding)
    fun add(name: String, embedding: Array<FloatArray>) = addList(ptr, name, embedding)
    fun remove(name: String) = remove(ptr, name)
    fun search(embedding: FloatArray, threshold: Float) = search(ptr, embedding, threshold)
    fun verify(name: String, embedding: FloatArray, threshold: Float) =
        verify(ptr, name, embedding, threshold)

    fun contains(name: String) = contains(ptr, name)
    fun numSpeakers() = numSpeakers(ptr)

    fun allSpeakerNames() = allSpeakerNames(ptr)

    private external fun new(dim: Int): Long
    private external fun delete(ptr: Long): Unit
    private external fun add(ptr: Long, name: String, embedding: FloatArray): Boolean
    private external fun addList(ptr: Long, name: String, embedding: Array<FloatArray>): Boolean
    private external fun remove(ptr: Long, name: String): Boolean
    private external fun search(ptr: Long, embedding: FloatArray, threshold: Float): String
    private external fun verify(
        ptr: Long,
        name: String,
        embedding: FloatArray,
        threshold: Float
    ): Boolean

    private external fun contains(ptr: Long, name: String): Boolean
    private external fun numSpeakers(ptr: Long): Int

    private external fun allSpeakerNames(ptr: Long): Array<String>

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}

// Please download the model file from
// https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models
// and put it inside the assets directory.
//
// Please don't put it in a subdirectory of assets
private val modelName = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

object SpeakerRecognition {
    var _extractor: SpeakerEmbeddingExtractor? = null
    var _manager: SpeakerEmbeddingManager? = null

    val extractor: SpeakerEmbeddingExtractor
        get() {
            return _extractor!!
        }

    val manager: SpeakerEmbeddingManager
        get() {
            return _manager!!
        }

    fun initExtractor(assetManager: AssetManager? = null) {
        synchronized(this) {
            if (_extractor != null) {
                return
            }
            Log.i(TAG, "Initializing speaker embedding extractor")

            _extractor = SpeakerEmbeddingExtractor(
                assetManager = assetManager,
                config = SpeakerEmbeddingExtractorConfig(
                    model = modelName,
                    numThreads = 2,
                    debug = false,
                    provider = "cpu",
                )
            )

            _manager = SpeakerEmbeddingManager(dim = _extractor!!.dim())
        }
    }
}
