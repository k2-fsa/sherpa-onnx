package com.k2fsa.sherpa.onnx.tts.engine.ui

import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import androidx.lifecycle.ViewModel
import com.k2fsa.sherpa.onnx.tts.engine.ui.error.showErrorDialog
import com.k2fsa.sherpa.onnx.tts.engine.utils.runOnUI

@Composable
fun ErrorHandler(vm: ImplViewModel, title: String? = null) {
    val context = LocalContext.current
    LaunchedEffect(key1 = vm.error) {
        vm.error?.let {
            context.showErrorDialog(it, title)
            vm.error = null
        }

    }
}

open class ImplViewModel() : ViewModel() {
    var error by mutableStateOf<Throwable?>(null)

    fun postError(t: Throwable) {
        runOnUI { error = t }
    }
}