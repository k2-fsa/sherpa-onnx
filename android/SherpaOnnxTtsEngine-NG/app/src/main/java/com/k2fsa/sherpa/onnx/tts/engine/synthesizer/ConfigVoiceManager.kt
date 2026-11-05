package com.k2fsa.sherpa.onnx.tts.engine.synthesizer

import android.util.Log
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.ConfigManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.Collections

object ConfigVoiceManager : MutableCollection<Voice> {
    private var speakers = mutableListOf<Voice>()
    fun speakers(): List<Voice> = speakers

    private val _flow by lazy { MutableStateFlow<List<Voice>>(emptyList()) }
    val flow: StateFlow<List<Voice>>
        get() = _flow.asStateFlow()

    private fun notifySpeakersChange() {
        Log.d(ConfigModelManager.TAG, "notifySpeakersChange: ${speakers.size}")
        ConfigManager.updateConfig(config = ConfigManager.config.copy(voices = speakers))
        _flow.value = speakers.toList()
    }

    @Synchronized
    fun load() {
        speakers = ConfigManager.config.voices.toMutableList()
        distinct()
        notifySpeakersChange()
    }

    fun move(from: Int, to: Int) {
        Collections.swap(speakers, from, to)
        notifySpeakersChange()
    }

    fun update(speaker: Voice) {
        val index = speakers.indexOfFirst { it.id == speaker.id }
        if (index != -1) {
            speakers[index] = speaker
            notifySpeakersChange()
        }
    }

    fun reset(list: List<Voice>) {
        speakers.clear()
        speakers.addAll(list)
        notifySpeakersChange()
    }

    fun distinct() {
        speakers = speakers.distinctBy { it.toString() }.toMutableList()
        notifySpeakersChange()
    }

    override fun add(element: Voice): Boolean {
        speakers.add(element)
        notifySpeakersChange()
        return true
    }

    override fun contains(element: Voice): Boolean {
        return speakers.contains(element)
    }

    override val size: Int
        get() = speakers.size

    override fun clear() {
        speakers.clear()
        notifySpeakersChange()
    }

    override fun addAll(elements: Collection<Voice>): Boolean {
        speakers.addAll(elements)
        notifySpeakersChange()
        return true
    }

    override fun isEmpty(): Boolean {
        return speakers.isEmpty()
    }

    override fun iterator(): MutableIterator<Voice> {
        return speakers.iterator()
    }

    override fun retainAll(elements: Collection<Voice>): Boolean {
        return speakers.retainAll(elements).apply {
            notifySpeakersChange()
        }
    }

    override fun removeAll(elements: Collection<Voice>): Boolean {
        return speakers.removeAll(elements).apply {
            notifySpeakersChange()
        }
    }

    override fun remove(element: Voice): Boolean {
        return speakers.remove(element).apply {
            notifySpeakersChange()
        }
    }

    override fun containsAll(elements: Collection<Voice>): Boolean {
        return speakers.containsAll(elements)
    }

}