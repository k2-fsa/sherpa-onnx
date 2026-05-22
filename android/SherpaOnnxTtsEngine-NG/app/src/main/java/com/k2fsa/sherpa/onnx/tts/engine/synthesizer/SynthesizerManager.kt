package com.k2fsa.sherpa.onnx.tts.engine.synthesizer

import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.tts.engine.conf.TtsConfig
import kotlin.math.max

object SynthesizerManager {
    const val TAG = "SynthesizerManager"

    private val cacheManager = SynthesizerCache()

    fun getTTS(cfg: OfflineTtsConfig): OfflineTts {
        val tts = (cacheManager.getById(cfg.model.vits.model) as Synthesizer?)?.tts
        tts?.let {
            Log.d(TAG, "getTTS (from cache): ${it.config}")
        }

        val model = cfg.model.copy(
            numThreads = max(1, TtsConfig.threadNum.value)
        )

        return tts ?: OfflineTts(config = cfg.copy(model = model)).run {
            cacheTTS(cfg.model.vits.model, this)
            this
        }
    }

    private fun cacheTTS(id: String, tts: OfflineTts) {
        Log.d(TAG, "cacheTTS: ${tts.config}")
        cacheManager.cache(id, Synthesizer(tts))
    }

    private class Synthesizer(val tts: OfflineTts) : ImplCache {
        override fun destroy() {
            tts.free()
        }

        override fun canDestroy(): Boolean = !tts.isRunning
    }
}