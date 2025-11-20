package com.k2fsa.sherpa.onnx.simulate.streaming.asr

import android.app.Application
import android.content.Context
import android.content.res.AssetManager
import android.os.Build
import android.util.Log
import com.k2fsa.sherpa.onnx.HomophoneReplacerConfig
import com.k2fsa.sherpa.onnx.OfflineRecognizer
import com.k2fsa.sherpa.onnx.OfflineRecognizerConfig
import com.k2fsa.sherpa.onnx.Vad
import com.k2fsa.sherpa.onnx.getOfflineModelConfig
import com.k2fsa.sherpa.onnx.getVadModelConfig
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.io.OutputStream


fun assetExists(assetManager: AssetManager, path: String): Boolean {
    val dir = path.substringBeforeLast('/', "")
    val fileName = path.substringAfterLast('/')

    val files = assetManager.list(dir) ?: return false
    return files.contains(fileName)
}

fun copyAssetToSdCard(path: String, context: Context): String {
    val targetRoot = context.filesDir
    val outFile = File(targetRoot, path)

    if (!assetExists(context.assets, path = path)) {
        // for context binary, if it is does not exist, we return a path
        // that can be written to
        outFile.parentFile?.mkdirs()
        Log.i(TAG, "$path does not exist, return ${outFile.absolutePath}")
        return outFile.absolutePath
    }

    if (outFile.exists()) {
        val assetSize = context.assets.open(path).use { it.available() }
        if (outFile.length() == assetSize.toLong()) {
            Log.i(TAG, "$targetRoot/$path already exists, skip copying, return $targetRoot/$path")

            return "$targetRoot/$path"
        }
    }

    outFile.parentFile?.mkdirs()

    context.assets.open(path).use { input: InputStream ->
        FileOutputStream(outFile).use { output: OutputStream ->
            input.copyTo(output)
        }
    }
    Log.i(TAG, "Copied $path to $targetRoot/$path")

    return outFile.absolutePath
}


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

    fun initOfflineRecognizer(context: Context, asrModelType: Int) {
        synchronized(this) {
            if (_recognizer != null) {
                return
            }
            Log.i(TAG, "Initializing sherpa-onnx offline recognizer")
            // Please change getOfflineModelConfig() to add new models
            // See https://k2-fsa.github.io/sherpa/onnx/pretrained_models/index.html
            // for a list of available models
            val asrRuleFsts: String?
            asrRuleFsts = null
            Log.i(TAG, "Select model type $asrModelType for ASR")

            val useHr = false
            val hr = HomophoneReplacerConfig(
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

            var assetManager: AssetManager? = context.assets

            if (config.modelConfig.provider == "qnn") {
                // We assume you have copied files like libQnnHtpV81Skel.so to jniLibs/arm64-v8a
                Log.i(TAG, "nativelibdir: ${context.applicationInfo.nativeLibraryDir}")

                // If we don't set the environment variable for ADSP_LIBRARY_PATH, we will see
                // the error code 1008 from qnn_interface.deviceCreate()
                // See also
                // https://workbench.aihub.qualcomm.com/docs/hub/faq.html#why-am-i-seeing-error-1008-when-trying-to-use-htp
                OfflineRecognizer.prependAdspLibraryPath(context.applicationInfo.nativeLibraryDir)

                // for qnn, we need to copy *.so files from assets folder to sd card
                if (config.modelConfig.senseVoice.qnnConfig.backendLib.isEmpty()) {
                    Log.e(TAG, "You should provide libQnnHtp.so for qnn")
                    throw IllegalArgumentException("You should provide libQnnHtp.so for qnn")
                }
                config.modelConfig.tokens = copyAssetToSdCard(config.modelConfig.tokens, context)

                config.modelConfig.senseVoice.model =
                    copyAssetToSdCard(config.modelConfig.senseVoice.model, context)

                config.modelConfig.senseVoice.qnnConfig.contextBinary = copyAssetToSdCard(
                    config.modelConfig.senseVoice.qnnConfig.contextBinary,
                    context
                )

                if (config.hr.lexicon.isNotEmpty()) {
                    config.hr.lexicon = copyAssetToSdCard(config.hr.lexicon, context)
                }

                if (config.hr.ruleFsts.isNotEmpty()) {
                    // it assumes there is only one fst. otherwise, you need to copy each fst separately
                    config.hr.ruleFsts = copyAssetToSdCard(config.hr.ruleFsts, context)
                }

                assetManager = null
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
