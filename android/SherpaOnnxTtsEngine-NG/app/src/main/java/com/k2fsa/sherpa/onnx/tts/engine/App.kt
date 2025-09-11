package com.k2fsa.sherpa.onnx.tts.engine

import android.app.Application
import com.drake.net.utils.withIO
import com.k2fsa.sherpa.onnx.tts.engine.utils.CompressorFactory
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

val app by lazy { App.instance }

class App : Application() {
    companion object {
        lateinit var instance: App
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

}