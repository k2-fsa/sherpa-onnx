package com.k2fsa.sherpa.onnx.tts.engine.ui.voices

import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.viewModelScope
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigVoiceManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Voice
import com.k2fsa.sherpa.onnx.tts.engine.ui.ImplViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class VoiceManagerViewModel : ImplViewModel() {
    var voices by mutableStateOf<List<Voice>>(emptyList())
    val selects = mutableStateListOf<Voice>()
    val listState = LazyListState()

    fun isSelected(voice: Voice) = selects.contains(voice)

    fun select(voice: Voice) {
        if (selects.contains(voice)) {
            selects.remove(voice)
        } else {
            selects.add(voice)
        }
    }

    fun selectAll() {
        selects.clear()
        selects.addAll(voices)
    }

    fun selectInvert() {
        val newSelects = voices.filter { !selects.contains(it) }
        selects.clear()
        selects.addAll(newSelects)
    }

    fun selectClear() {
        selects.clear()
    }

    fun load() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                ConfigVoiceManager.load()
                if (ConfigModelManager.models().isEmpty())
                    ConfigModelManager.load()
            }.onSuccess {
                ConfigVoiceManager.flow.collectLatest {
                    selects.clear()
                    voices = it
                }
            }.onFailure {
                postError(it)
            }
        }
    }

    fun move(from: Int, to: Int) {
        ConfigVoiceManager.move(from, to)
    }

    fun delete(voice: List<Voice>) {
        ConfigVoiceManager.removeAll(voice)
    }

    fun addVoice(voice: Voice) {
        ConfigVoiceManager.add(voice)

    }

    fun sortByName() {
        ConfigVoiceManager.reset(voices.sortedBy { it.name })
    }

    fun sortByModel() {
        ConfigVoiceManager.reset(voices.sortedBy { it.model + it.id })
    }

    fun updateVoice(voice: Voice) {
        ConfigVoiceManager.update(voice)
    }

    fun isModelAvailable(voice: Voice): Boolean {
        return ConfigModelManager.models().any { it.id == voice.model }
    }

    fun unselect(voice: Voice) {
        selects.remove(voice)
    }
}