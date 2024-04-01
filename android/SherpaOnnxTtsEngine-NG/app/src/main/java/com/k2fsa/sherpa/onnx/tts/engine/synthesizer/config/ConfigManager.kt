package com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config

import com.charleskorn.kaml.decodeFromStream
import com.k2fsa.sherpa.onnx.tts.engine.AppConst
import com.k2fsa.sherpa.onnx.tts.engine.FileConst.configPath
import kotlinx.serialization.encodeToString
import java.io.InputStream

object ConfigManager : ImplYamlConfig<Config>(configPath, { Config() }) {
    const val TAG = "ConfigManager"

    override fun encode(o: Config): String {
        return AppConst.yaml.encodeToString(o)
    }

    override fun decode(ins: InputStream): Config {
        return AppConst.yaml.decodeFromStream(ins)
    }

}