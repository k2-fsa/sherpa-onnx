package com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config

import kotlinx.serialization.Serializable

@Serializable
data class Voice(val model: String = "", val id: Int = 0, val name: String = "") {
    companion object {
        val EMPTY = Voice()

        fun from(s: String): Voice {
            val i = s.indexOfLast { it == '_' }
            val id = if (i != -1) s.substring(i + 1) else ""
            val model = if (i != -1) s.substring(0, i) else ""
            return if (id.isBlank() || model.isBlank()) EMPTY
            else try {
                Voice(model = model, id = id.toInt())
            } catch (_: NumberFormatException) {
                EMPTY
            }
        }
    }

    override fun toString(): String = "${model}_${id}"

    fun contains(voice: Voice): Boolean = voice.toString() == this.toString()
}