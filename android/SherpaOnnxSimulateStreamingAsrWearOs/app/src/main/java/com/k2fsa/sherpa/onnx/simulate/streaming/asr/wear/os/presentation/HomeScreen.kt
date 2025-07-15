package com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.TimeText
import com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation.theme.SherpaOnnxSimulateStreamingAsrWearOsTheme

@Composable
fun HomeScreen() {
    SherpaOnnxSimulateStreamingAsrWearOsTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            TimeText()
            Greeting(greetingName = "k2")
        }
    }

}