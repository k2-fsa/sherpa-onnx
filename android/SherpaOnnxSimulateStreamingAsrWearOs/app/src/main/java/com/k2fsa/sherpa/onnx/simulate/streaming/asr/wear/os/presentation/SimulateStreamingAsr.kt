package com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation

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
            val asrModelType = 39
            val asrRuleFsts: String?
            asrRuleFsts = null
            Log.i(TAG, "Select model type $asrModelType for ASR")

            val useHr = false
            val hr = HomophoneReplacerConfig(
                // Used only when useHr is true
                // Please download the following 3 files from
                // https://github.com/k2-fsa/sherpa-onnx/releases/tag/hr-files
                //
                // dict and lexicon.txt can be shared by different apps
                //
                // replace.fst is specific for an app
                dictDir = "dict",
                lexicon = "lexicon.txt",
                ruleFsts = "replace.fst",
            )

            val config = OfflineRecognizerConfig(
                modelConfig = getOfflineModelConfig(type = asrModelType)!!,
            )

            if (config.modelConfig.numThreads == 1) {
                config.modelConfig.numThreads = 2
            }
            config.modelConfig.debug = true

            if (asrRuleFsts != null) {
                config.ruleFsts = asrRuleFsts
            }

            if (useHr) {
                if (hr.dictDir.isNotEmpty() && hr.dictDir.first() != '/') {
                    // We need to copy it from the assets directory to some path
                    val newDir = copyDataDir(hr.dictDir, application)
                    hr.dictDir = "$newDir/${hr.dictDir}"
                }
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

    private fun copyDataDir(dataDir: String, application: Application): String {
        Log.i(TAG, "data dir is $dataDir")
        copyAssets(dataDir, application)

        val newDataDir = application.getExternalFilesDir(null)!!.absolutePath
        Log.i(TAG, "newDataDir: $newDataDir")
        return newDataDir
    }

    private fun copyAssets(path: String, application: Application) {
        val assets: Array<String>?
        try {
            assets = application.assets.list(path)
            if (assets!!.isEmpty()) {
                copyFile(path, application)
            } else {
                val fullPath = "${application.getExternalFilesDir(null)}/$path"
                val dir = File(fullPath)
                dir.mkdirs()
                for (asset in assets.iterator()) {
                    val p: String = if (path == "") "" else "$path/"
                    copyAssets(p + asset, application)
                }
            }
        } catch (ex: IOException) {
            Log.e(TAG, "Failed to copy $path. $ex")
        }
    }

    private fun copyFile(filename: String, application: Application) {
        try {
            val istream = application.assets.open(filename)
            val newFilename = application.getExternalFilesDir(null).toString() + "/" + filename
            val ostream = FileOutputStream(newFilename)
            // Log.i(TAG, "Copying $filename to $newFilename")
            val buffer = ByteArray(1024)
            var read = 0
            while (read != -1) {
                ostream.write(buffer, 0, read)
                read = istream.read(buffer)
            }
            istream.close()
            ostream.flush()
            ostream.close()
        } catch (ex: Exception) {
            Log.e(TAG, "Failed to copy $filename, $ex")
        }
    }
}