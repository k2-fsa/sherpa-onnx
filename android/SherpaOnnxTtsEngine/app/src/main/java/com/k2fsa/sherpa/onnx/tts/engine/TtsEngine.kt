package com.k2fsa.sherpa.onnx.tts.engine

import android.content.Context
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
    private var ruleFars: String? = null
    private var lexicon: String? = null
    private var dataDir: String? = null
    private var assets: AssetManager? = null

    init {
        // The purpose of such a design is to make the CI test easier
        // Please see
        // https://github.com/k2-fsa/sherpa-onnx/blob/master/scripts/apk/generate-tts-apk-script.py
        modelDir = null
        modelName = null
        ruleFsts = null
        ruleFars = null
        lexicon = null
        dataDir = null
        lang = null

        // Please enable one and only one of the examples below

        // Example 1:
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-vctk.tar.bz2
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
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-icefall-zh-aishell3.tar.bz2
        // modelDir = "vits-icefall-zh-aishell3"
        // modelName = "model.onnx"
        // ruleFsts = "vits-icefall-zh-aishell3/phone.fst,vits-icefall-zh-aishell3/date.fst,vits-icefall-zh-aishell3/number.fst,vits-icefall-zh-aishell3/new_heteronym.fst"
        // ruleFars = "vits-icefall-zh-aishell3/rule.far"
        // lexicon = "lexicon.txt"
        // lang = "zho"

        // Example 4:
        // https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-coqui-de-css10.tar.bz2
        // This model does not need lexicon or dataDir
        // modelDir = "vits-coqui-de-css10"
        // modelName = "model.onnx"
        // lang = "deu"
    }


    fun createTts(context: Context) {
        Log.i(TAG, "Init Next-gen Kaldi TTS")
        if (tts == null) {
            initTts(context)
        }
    }

    private fun initTts(context: Context) {
        assets = context.assets

        if (dataDir != null) {
            val newDir = copyDataDir(context, modelDir!!)
            modelDir = newDir + "/" + modelDir
            dataDir = newDir + "/" + dataDir
            assets = null
        }

        val config = getOfflineTtsConfig(
            modelDir = modelDir!!, modelName = modelName!!, lexicon = lexicon ?: "",
            dataDir = dataDir ?: "",
            ruleFsts = ruleFsts ?: "",
            ruleFars = ruleFars ?: ""
        )!!

        tts = OfflineTts(assetManager = assets, config = config)
    }


    private fun copyDataDir(context: Context, dataDir: String): String {
        println("data dir is $dataDir")
        copyAssets(context, dataDir)

        val newDataDir = context.getExternalFilesDir(null)!!.absolutePath
        println("newDataDir: $newDataDir")
        return newDataDir
    }

    private fun copyAssets(context: Context, path: String) {
        val assets: Array<String>?
        try {
            assets = context.assets.list(path)
            if (assets!!.isEmpty()) {
                copyFile(context, path)
            } else {
                val fullPath = "${context.getExternalFilesDir(null)}/$path"
                val dir = File(fullPath)
                dir.mkdirs()
                for (asset in assets.iterator()) {
                    val p: String = if (path == "") "" else path + "/"
                    copyAssets(context, p + asset)
                }
            }
        } catch (ex: IOException) {
            Log.e(TAG, "Failed to copy $path. ${ex.toString()}")
        }
    }

    private fun copyFile(context: Context, filename: String) {
        try {
            val istream = context.assets.open(filename)
            val newFilename = context.getExternalFilesDir(null).toString() + "/" + filename
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
