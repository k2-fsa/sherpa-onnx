package com.k2fsa.sherpa.onnx.speaker.identification.screens

import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun HelpScreen() {
    Box(modifier= Modifier.fillMaxSize()) {
        Column(
            modifier = Modifier.padding(16.dp)
        ) {
            Text("Please see http://github.com/k2-fsa/sherpa-onnx ")
            Spacer(modifier = Modifier.height(16.dp))
            Text("https://github.com/k2-fsa/sherpa-onnx/releases/tag/speaker-recongition-models")
            Spacer(modifier = Modifier.height(16.dp))
            Text("https://k2-fsa.github.io/sherpa/social-groups.html")
            Spacer(modifier = Modifier.height(16.dp))
            Text("Everything is open-sourced!")
        }
    }
}
