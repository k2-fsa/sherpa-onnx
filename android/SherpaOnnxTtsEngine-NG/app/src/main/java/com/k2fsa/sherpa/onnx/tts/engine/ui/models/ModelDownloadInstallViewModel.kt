package com.k2fsa.sherpa.onnx.tts.engine.ui.models

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.drake.net.Get
import com.k2fsa.sherpa.onnx.tts.engine.AppConst
import com.k2fsa.sherpa.onnx.tts.engine.GithubRelease
import com.k2fsa.sherpa.onnx.tts.engine.synthesizer.ConfigModelManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

class ModelDownloadInstallViewModel : ViewModel() {
    var error by mutableStateOf("")
    val modelList = mutableStateListOf<GithubRelease.Asset>()
//    val checkedModels = mutableStateListOf<GithubRelease.Asset>()

    fun load() {
        error = ""
        modelList.clear()
        viewModelScope.launch(Dispatchers.IO) {
            runCatching {
                val str =
                    Get<String>("https://api.github.com/repos/k2-fsa/sherpa-onnx/releases/tags/tts-models").await()
                val release: GithubRelease = AppConst.jsonBuilder.decodeFromString(str)
                val addedModels = ConfigModelManager.models()
                modelList.addAll(release.assets.filter { asset ->
                    // contains only tar.bz2 and not added
                    val ext = ".tar.bz2"
                    asset.name.endsWith(ext) &&
                            addedModels.find { it.id == asset.name.removeSuffix(ext) } == null
                })
            }.onFailure {
                error = it.message ?: "Unknown error"
            }
        }
    }
}