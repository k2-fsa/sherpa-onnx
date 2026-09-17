package com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config

import kotlinx.serialization.Serializable

@Serializable
data class Model(
    val id: String,
    val name: String = id,
    val onnx: String,
    val dataDir: String,

    val lexicon: String,
    val ruleFsts: String,
    val tokens: String,

    val lang: String,
) {
    companion object {
        val EMPTY = Model(
            id = "",
            onnx = "",
            dataDir = "",
            lexicon = "",
            ruleFsts = "",
            tokens = "",
            lang = ""
        )
    }
}