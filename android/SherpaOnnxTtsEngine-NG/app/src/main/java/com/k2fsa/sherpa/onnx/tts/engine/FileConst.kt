package com.k2fsa.sherpa.onnx.tts.engine

object FileConst {
    var modelDir: String =
        app.getExternalFilesDir("model")!!.absolutePath

    var cacheModelDir =
        app.externalCacheDir!!.resolve("model").absolutePath
    val cacheDownloadDir = app.externalCacheDir!!.resolve("download").absolutePath

    var configPath: String = "$modelDir/config.yaml"
    var sampleTextPath: String = "$modelDir/sampleText.yaml"
}