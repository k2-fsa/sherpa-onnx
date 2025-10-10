package com.k2fsa.sherpa.onnx.simulate.streaming.asr

import android.app.Application
import android.content.res.AssetManager
import android.util.Log
import com.k2fsa.sherpa.onnx.HomophoneReplacerConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.getOfflineModelConfig
import com.k2fsa.sherpa.onnx.getVadModelConfig
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object SimulateStreamingAsr {
    private var _recognizer: OfflineRecognizer? = null
    val recognizer: OfflineRecognizer
        get() {
            return _recognizer!!
        }

    private var _vad: Vad? = null
    val vad: Vad
        get() {
            return _vad!!
        }

    fun initOfflineRecognizer(assetManager: AssetManager? = null, application: Application) {
        synchronized(this) {
            if (_recognizer != null) {
                return
            }
            Log.i(TAG, "Initializing sherpa-onnx offline recognizer")
            // Please change getOfflineModelConfig() to add new models
            // See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
            // for a list of available models
            val asrModelType = 15
            val asrRuleFsts: String?
            asrRuleFsts = null
            Log.i(TAG, "Select model type $asrModelType for ASR")

            val useHr = false
            val hr =  HomophoneReplacerConfig(
                // Used only when useHr is true
                // Please download the following 2 files from
                // https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files
                //
                // lexicon.txt can be shared by different apps
                //
                // replace.fst is specific for an app
                lexicon = "lexicon.txt",
                ruleFsts = "replace.fst",
            )

            val config = OfflineRecognizerConfig(
                modelConfig = getOfflineModelConfig(type = asrModelType)!!,
            )

            if (config.modelConfig.numThreads == 1) {
                config.modelConfig.numThreads = 2
            }

            if (asrRuleFsts != null) {
                config.ruleFsts = asrRuleFsts
            }

            if (useHr) {
                config.hr = hr
            }

            _recognizer = OfflineRecognizer(
                assetManager = assetManager,
                config = config,
            )

            Log.i(TAG, "sherpa-onnx offline recognizer initialized")
        }
    }

    fun initVad(assetManager: AssetManager? = null) {
        if (_vad != null) {
            return
        }
        val type = 0
        Log.i(TAG, "Select VAD model type $type")
        val config = getVadModelConfig(type)

        _vad = Vad(
            assetManager = assetManager,
            config = config!!,
        )
        Log.i(TAG, "sherpa-onnx vad initialized")
    }
}
