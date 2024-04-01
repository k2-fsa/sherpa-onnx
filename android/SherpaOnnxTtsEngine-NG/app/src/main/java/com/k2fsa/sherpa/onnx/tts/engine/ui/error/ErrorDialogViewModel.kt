package com.k2fsa.sherpa.onnx.tts.engine.ui.error

import androidx.lifecycle.ViewModel

class ErrorDialogViewModel : ViewModel() {
    internal val throwableList = mutableMapOf<String, Throwable>()
}