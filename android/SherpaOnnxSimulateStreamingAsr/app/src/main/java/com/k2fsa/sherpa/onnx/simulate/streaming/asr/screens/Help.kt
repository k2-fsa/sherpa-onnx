package com.k2fsa.sherpa.onnx.simulate.streaming.asr.screens

import androidx.compose.runtime.Composable
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun HelpScreen() {
    Box(modifier = Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.padding(8.dp)
        ) {
            Text(
                "This app uses a non-streaming ASR model together with silero-vad " +
                        "for streaming/real-time speech recognition. ",
                fontSize=10.sp
            )
            Spacer(modifier = Modifier.height(10.dp))
            Text("Please see http://github.com/k2-fsa/sherpa-onnx ")

            Spacer(modifier = Modifier.height(10.dp))
            Text("Everything is open-sourced!", fontSize = 20.sp)
        }
    }
}
