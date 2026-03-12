package com.k2fsa.sherpa.onnx

class OfflineStream(var ptr: Long) {
    fun acceptWaveform(samples: FloatArray, sampleRate: Int) =
        acceptWaveform(ptr, samples, sampleRate)

    fun setOption(key: String, value: String) = setOption(ptr, key, value)

    fun getOption(key: String): String = getOption(ptr, key)

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    fun release() = finalize()

    fun use(block: (OfflineStream) -> Unit) {
        try {
            block(this)
        } finally {
            release()
        }
    }

    private external fun acceptWaveform(ptr: Long, samples: FloatArray, sampleRate: Int)
    private external fun setOption(ptr: Long, key: String, value: String)
    private external fun getOption(ptr: Long, key: String): String
    private external fun delete(ptr: Long)

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
