package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.foundation.lazy.LazyListState
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.viewModelScope
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ModelPackageManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import com.k2fsa.sherpa.onnx.tts.engine.ui.ImplViewModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.util.Collections

class ModelManagerViewModel : ImplViewModel() {
    internal val models = mutableStateOf<List<Model>>(emptyList())
    internal val selectedModels = mutableStateListOf<Model>()
    internal val listState by lazy { LazyListState() }

    override fun onCleared() {
        super.onCleared()
        models.value = emptyList()
        selectedModels.clear()
    }

    fun load() {
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                ConfigModelManager.load()
            }.onSuccess {
                ConfigModelManager.modelsFlow.collectLatest {
                    selectedModels.clear()
                    models.value = it
                }
            }.onFailure {
                postError(it)
            }
        }
    }

    fun moveModel(from: Int, to: Int) {
        val list = ConfigModelManager.models().toMutableList()
        Collections.swap(list, from, to)
        ConfigModelManager.updateModels(list)
    }

    fun deleteModel(model: Model) {
        ConfigModelManager.removeModel(model)
    }

    fun setLanguagesForSelectedModels(lang: String) {
        val list = ConfigModelManager.models().toMutableList()
        list.forEachIndexed { index, model ->
            if (selectedModels.find { it.id == model.id } != null) {
                list[index] = model.copy(lang = lang)
            }
        }
        ConfigModelManager.updateModels(list)
    }

    fun selectAll() {
        selectedModels.clear()
        selectedModels.addAll(models.value)
    }

    fun selectInvert() {
        ConfigModelManager.models().filter { !selectedModels.contains(it) }.let {
            selectedModels.clear()
            selectedModels.addAll(it)
        }
    }

    fun clearSelect() {
        selectedModels.clear()
    }

    fun deleteModels(models: List<Model>, deleteFile: Boolean) {
        if (deleteFile)
            models.forEach {
                ModelPackageManager.deleteModel(it.id)
            }
        ConfigModelManager.removeModel(*models.toTypedArray())
    }
}