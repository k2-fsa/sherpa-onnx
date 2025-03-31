package com.k2fsa.sherpa.onnx.tts.engine.synthesizer

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import com.k2fsa.sherpa.onnx.tts.engine.FileConst
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.ConfigManager
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.config.Model
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.io.File

object ConfigModelManager {
    const val TAG = "ModelManager"

    private val models = mutableListOf<Model>()

    fun models(): List<Model> = models

    private val _modelsFlow by lazy { MutableStateFlow<List<Model>>(emptyList()) }
    val modelsFlow: StateFlow<List<Model>>
        get() = _modelsFlow.asStateFlow()


    private fun notifyModelsChange() {
        Log.d(TAG, "notifyModelsChange: ${models.size}")
        _modelsFlow.value = models.toList()
    }

    @Synchronized
    fun load() {
        models.addAll(ConfigManager.config.models)
        instinctModels()
        notifyModelsChange()
    }

    fun languages(): List<String> {
        return models.distinctBy { it.lang }.map { it.lang }
    }

    fun defaultLanguage(): String {
        return models.firstOrNull()?.lang ?: "en"
    }

    // 去重models
    @Synchronized
    private fun instinctModels() {
        val list = models.distinctBy { it.id }
        Log.d(TAG, "deduplicateModels: ${list.size}")
        models.clear()
        models.addAll(list)
        ConfigManager.updateConfig(ConfigManager.config.copy(models = models))
    }

    @Synchronized
    fun removeModel(vararg model: Model) {
        models.removeAll(model.toSet())
        ConfigManager.updateConfig(ConfigManager.config.copy(models = models))
        notifyModelsChange()
    }

    @Synchronized
    fun addModel(vararg model: Model) {
        model.forEach { m ->
            if (!models.any { it.id == m.id })
                models.add(m)
        }

        ConfigManager.updateConfig(ConfigManager.config.copy(models = models))
        notifyModelsChange()
    }

    fun updateModels(model: Model) {
        models.indexOfFirst { it.id == model.id }.takeIf { it != -1 }?.let {
            models[it] = model.copy()
            ConfigManager.updateConfig(ConfigManager.config.copy(models = models))
            notifyModelsChange()
        }
    }

    @Synchronized
    fun updateModels(model: List<Model>) {
        models.clear()
        models.addAll(model)
        ConfigManager.updateConfig(ConfigManager.config.copy(models = models))
        notifyModelsChange()
    }

    fun getNotAddedModels(context: Context): List<Model> {
        val addedIds = models.map { it.id }
        return analyzeToModels().filter { it.id !in addedIds }
    }

    fun analyzeToModel(dir: File): Model? {
        Log.d(TAG, "load model: ${dir.name}")
        val onnx = dir.listFiles { _, name -> name.endsWith(".onnx") }
            ?.run { if (isNotEmpty()) first() else null }
            ?: return null

        val dataDir = dir.resolve("espeak-ng-data").takeIf { it.exists() }

        return Model(
            id = dir.name,
            onnx = dir.name + "/" + onnx.name,
            lexicon = if (dataDir == null) "${dir.name}/lexicon.txt" else "",
            ruleFsts = if (dataDir == null) "${dir.name}/rule.fst" else "",
            tokens = "${dir.name}/tokens.txt",
            dataDir = dataDir?.run { "${dir.name}/espeak-ng-data" } ?: "",
            lang = "en-US",
        )
    }

    // 根据文件目录结构获取模型列表
    fun analyzeToModels(): List<Model> {
        Log.d(TAG, "modelPath: ${FileConst.modelDir}")
        val list = mutableListOf<Model>()
        File(FileConst.modelDir).listFiles()!!.forEach { dir ->
            if (dir.isDirectory)
                analyzeToModel(dir)?.let {
                    list.add(it)
                }
        }

        return list
    }

    fun Model.toOfflineTtsConfig(root: String = FileConst.modelDir): OfflineTtsConfig {
        fun format(str: String): String {
            return if (str.isBlank()) "" else "$root/$str"
        }

        return OfflineTtsConfig(
            model = OfflineTtsModelConfig(
                vits = OfflineTtsVitsModelConfig(
                    model = format(onnx),
                    lexicon = format(lexicon),
                    tokens = format(tokens),
                    dataDir = format(dataDir),
                ),
                debug = true,
                provider = "cpu",
            ),
            ruleFsts = format(ruleFsts),
        )
    }
}