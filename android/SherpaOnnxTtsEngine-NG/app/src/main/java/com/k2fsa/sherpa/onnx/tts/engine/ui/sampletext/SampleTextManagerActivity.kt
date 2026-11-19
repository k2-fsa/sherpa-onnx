package com.k2fsa.sherpa.onnx.tts.engine.ui.sampletext

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import com.k2fsa.sherpa.onnx.tts.engine.ui.theme.SherpaOnnxTtsEngineTheme

class SampleTextManagerActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            SherpaOnnxTtsEngineTheme {
                SampleTextManagerScreen()
            }
        }
    }
}