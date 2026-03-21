package com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config

import com.charleskorn.kaml.decodeFromStream
import com.k2fsa.sherpa.onnx.tts.engine.AppConst
import com.k2fsa.sherpa.onnx.tts.engine.FileConst
import kotlinx.serialization.encodeToString
import java.io.InputStream

typealias SampleTextMap = Map<String, List<String>>

object SampleTextConfig :
    ImplYamlConfig<SampleTextMap>(FileConst.sampleTextPath, { mapOf() }) {
    override fun decode(ins: InputStream): SampleTextMap {
        return AppConst.yaml.decodeFromStream(ins)
    }

    override fun encode(o: SampleTextMap): String {
        return AppConst.yaml.encodeToString(o)
    }

    operator fun get(code: String): List<String>? {
        return config[code]
    }

    operator fun set(code: String, list: List<String>) {
        updateConfig(config.toMutableMap().apply {
            this[code] = list
        })
    }

    fun remove(code: String) {
        updateConfig(config.toMutableMap().apply {
            this.remove(code)
        })
    }
}