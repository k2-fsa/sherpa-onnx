package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OnlineSpeechDenoiserConfig(
    var model: OfflineSpeechDenoiserModelConfig = OfflineSpeechDenoiserModelConfig(),
)

class OnlineSpeechDenoiser(
    assetManager: AssetManager? = null,
    config: OnlineSpeechDenoiserConfig,
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

    fun run(samples: FloatArray, sampleRate: Int) = run(ptr, samples, sampleRate)

    fun flush() = flush(ptr)

    fun reset() = reset(ptr)

    val sampleRate
      get() = getSampleRate(ptr)

    val frameShiftInSamples
      get() = getFrameShiftInSamples(ptr)

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OnlineSpeechDenoiserConfig,
    ): Long

    private external fun newFromFile(
        config: OnlineSpeechDenoiserConfig,
    ): Long

    private external fun delete(ptr: Long)

    private external fun run(ptr: Long, samples: FloatArray, sampleRate: Int): DenoisedAudio

    private external fun flush(ptr: Long): DenoisedAudio

    private external fun reset(ptr: Long)

    private external fun getSampleRate(ptr: Long): Int

    private external fun getFrameShiftInSamples(ptr: Long): Int

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
