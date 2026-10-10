package com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config

import android.util.Log
import com.k2fsa.sherpa.onnx.tts.engine.app
import com.k2fsa.sherpa.onnx.tts.engine.utils.longToast
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import java.io.File
import java.io.FileNotFoundException
import java.io.InputStream

open class ImplYamlConfig<T>(val filePath: String, val factory: () -> T) {
    open fun decode(ins: InputStream): T {
        throw NotImplementedError("")
    }

    open fun encode(o: T): String {
        throw NotImplementedError("")
    }

    private var mConfig: T? = null

    val config
        get() = mConfig ?: load().run { mConfig!! }

    fun updateConfig(config: T) {
        mConfig = config
        write(config = config)

        _configFlow.tryEmit(mConfig!!)
    }

    fun load() {
        mConfig = read()
        _configFlow.tryEmit(mConfig!!)
    }

    private val _configFlow by lazy {
        MutableStateFlow(config)
    }

    val configFlow: Flow<T>
        get() = _configFlow

    private fun read(path: String = filePath): T {
        val file = File(path)

        try {
            file.inputStream().use {
                return decode(it)
            }

        } catch (e: FileNotFoundException) {
            Log.e(ConfigManager.TAG, "readConfig: ", e)

            val obj = factory()
            write(config = obj)

            return obj
        }
    }

    private fun write(path: String = filePath, config: T) {
        val file = File(path)

        file.writeText(encode(config))
    }
}