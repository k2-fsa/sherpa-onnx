package com.k2fsa.sherpa.onnx.audio.tagging

import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.AudioTagging
import com.k2fsa.sherpa.onnx.getAudioTaggingConfig


object Tagger {
    private var _tagger: AudioTagging? = null
    val tagger: AudioTagging
        get() {
            return _tagger!!
        }

    fun initTagger(assetManager: AssetManager? = null, numThreads: Int = 1) {
        synchronized(this) {
            if (_tagger != null) {
                return
            }

            Log.i("sherpa-onnx", "Initializing audio tagger")
            val config = getAudioTaggingConfig(type = 0, numThreads = numThreads)!!
            _tagger = AudioTagging(assetManager, config)
        }
    }
}