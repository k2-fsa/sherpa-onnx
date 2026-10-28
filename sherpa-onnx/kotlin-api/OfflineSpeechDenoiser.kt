package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

data class OfflineSpeechDenoiserGtcrnModelConfig(
    var model: String = "",
)

data class OfflineSpeechDenoiserDpdfNetModelConfig(
    var model: String = "",
    var attenuationLimitDb: Float = 0.0f,
)

// VoxLock keeps the voice of one speaker (the caller) and removes the other
// voices. The model is from https://github.com/arbazsiddiqui/voxlock
data class OfflineSpeechDenoiserVoxLockModelConfig(
    var enrollModel: String = "",
    var chunkModel: String = "",
    // Optional. A wave file with at least 3 seconds of the caller's voice.
    // From a longer file, the loudest 3 seconds within its first 30 seconds
    // are used. A file that is shorter or cannot be read is ignored, as if it
    // were empty. If empty or ignored, the voice sample is taken from the
    // input audio: the offline denoiser uses the loudest 3 seconds in the
    // first 15 seconds of the input, and the online denoiser passes audio
    // through unchanged until 3 seconds of speech have been heard, then uses
    // that speech.
    var voiceSample: String = "",
)

data class OfflineSpeechDenoiserModelConfig(
    var gtcrn: OfflineSpeechDenoiserGtcrnModelConfig = OfflineSpeechDenoiserGtcrnModelConfig(),
    var dpdfnet: OfflineSpeechDenoiserDpdfNetModelConfig = OfflineSpeechDenoiserDpdfNetModelConfig(),
    var voxlock: OfflineSpeechDenoiserVoxLockModelConfig =
        OfflineSpeechDenoiserVoxLockModelConfig(),
    var numThreads: Int = 1,
    var debug: Boolean = false,
    var provider: String = "cpu",
)

data class OfflineSpeechDenoiserConfig(
    var model: OfflineSpeechDenoiserModelConfig = OfflineSpeechDenoiserModelConfig(),
)

class OfflineSpeechDenoiser(
    assetManager: AssetManager? = null,
    config: OfflineSpeechDenoiserConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
        require(ptr != 0L) {
            "Invalid OfflineSpeechDenoiserConfig: failed to create native OfflineSpeechDenoiser"
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

    val sampleRate
      get() = getSampleRate(ptr)

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: OfflineSpeechDenoiserConfig,
    ): Long

    private external fun newFromFile(
        config: OfflineSpeechDenoiserConfig,
    ): Long

    private external fun delete(ptr: Long)

    private external fun run(ptr: Long, samples: FloatArray, sampleRate: Int): DenoisedAudio

    private external fun getSampleRate(ptr: Long): Int

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
