package com.k2fsa.sherpa.onnx.tts.engine

import android.app.Application
import android.content.res.AssetManager
import android.util.Log
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import com.k2fsa.sherpa.onnx.*
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

object TtsEngine {
    var tts: OfflineTts? = null

    // https://en.wikipedia.org/wiki/ISO_639-3
    // Example:
    // eng for English,
    // deu for German
    // cmn for Mandarin
    var lang: String? = null



    val speedState: MutableState<Float> = mutableStateOf(1.0F)
    val speakerIdState: MutableState<Int> = mutableStateOf(0)

    var speed: Float
        get() = speedState.value
        set(value) {
            speedState.value = value
        }

    var speakerId: Int
        get() = speakerIdState.value
        set(value) {
            speakerIdState.value = value
        }

    private var modelDir: String? = null
    private var modelName: String? = null
    private var ruleFsts: String? = null
    private var lexicon: String? = null
    private var dataDir: String? = null
    private var assets: AssetManager? = null

    private var application: Application? = null

    fun createTts(application: Application) {
        Log.i(TAG, "Init Next-gen Kaldi TTS")
        if (tts == null) {
            this.application = application
            initTts()
        }
    }

    private fun initTts() {
        assets = application?.assets

        // The purpose of such a design is to make the CI test easier
        // Please see
        // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/apk/generate-tts-apk-script.py
        modelDir = null
        modelName = null
        ruleFsts = null
        lexicon = null
        dataDir = null
        lang = null

        // Please enable one and only one of the examples below

        // Example 1:
        // modelDir = "vits-vctk"
        // modelName = "vits-vctk.onnx"
        // lexicon = "lexicon.txt"
        // lang = "eng"

        // Example 2:
        // https://github.com/k2-fsa/sherpa-onnx/releases/tag/tts-models
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-amy-low.tar.bz2
        // modelDir = "vits-piper-en_US-amy-low"
        // modelName = "en_US-amy-low.onnx"
        // dataDir = "vits-piper-en_US-amy-low/espeak-ng-data"
        // lang = "eng"

        // Example 3:
        // modelDir = "vits-zh-aishell3"
        // modelName = "vits-aishell3.onnx"
        // ruleFsts = "vits-zh-aishell3/rule.fst"
        // lexcion = "lexicon.txt"
        // lang = "zho"

        if (dataDir != null) {
            val newDir = copyDataDir(modelDir!!)
            modelDir = newDir + "/" + modelDir
            dataDir = newDir + "/" + dataDir
            assets = null
        }

        val config = getOfflineTtsConfig(
            modelDir = modelDir!!, modelName = modelName!!, lexicon = lexicon ?: "",
            dataDir = dataDir ?: "",
            ruleFsts = ruleFsts ?: ""
        )!!

        tts = OfflineTts(assetManager = assets, config = config)
    }


    private fun copyDataDir(dataDir: String): String {
        println("data dir is $dataDir")
        copyAssets(dataDir)

        val newDataDir = application!!.getExternalFilesDir(null)!!.absolutePath
        println("newDataDir: $newDataDir")
        return newDataDir
    }

    private fun copyAssets(path: String) {
        val assets: Array<String>?
        try {
            assets = application!!.assets.list(path)
            if (assets!!.isEmpty()) {
                copyFile(path)
            } else {
                val fullPath = "${application!!.getExternalFilesDir(null)}/$path"
                val dir = File(fullPath)
                dir.mkdirs()
                for (asset in assets.iterator()) {
                    val p: String = if (path == "") "" else path + "/"
                    copyAssets(p + asset)
                }
            }
        } catch (ex: IOException) {
            Log.e(TAG, "Failed to copy $path. ${ex.toString()}")
        }
    }

    private fun copyFile(filename: String) {
        try {
            val istream = application!!.assets.open(filename)
            val newFilename = application!!.getExternalFilesDir(null).toString() + "/" + filename
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
            Log.e(TAG, "Failed to copy $filename, ${ex.toString()}")
        }
    }
}
