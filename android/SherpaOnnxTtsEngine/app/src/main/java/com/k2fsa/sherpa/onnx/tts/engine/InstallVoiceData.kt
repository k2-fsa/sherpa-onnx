package com.k2fsa.sherpa.onnx.tts.engine

import android.app.Activity
import android.os.Bundle
import android.view.Window

class InstallVoiceData : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        super.onCreate(savedInstanceState)
    }
}