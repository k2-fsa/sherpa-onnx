package com.k2fsa.sherpa.onnx.tts.engine.ui.sampletext

import androidx.compose.runtime.mutableStateListOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.SampleTextConfig
import kotlinx.coroutines.launch

class SampleTextMangerViewModel : ViewModel() {
    val languages = mutableStateListOf<String>()

    init {
        load()
    }

    fun load() {
        viewModelScope.launch {
            SampleTextConfig.configFlow.collect { data ->
                val l = data.toList().map { it.first }
                languages.clear()
                languages.addAll(l)

            }
        }
    }

    fun getList(code: String): List<String>? {
        return SampleTextConfig[code]
    }

    fun updateList(code: String, list: List<String>) {
        SampleTextConfig[code] = list
    }

    fun addLanguage(code: String) {
        SampleTextConfig[code] = emptyList()
    }

    fun removeLanguage(code: String) {
        SampleTextConfig.remove(code)
    }
}