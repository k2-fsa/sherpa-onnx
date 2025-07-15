package com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation

import android.app.Activity
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.TimeText
import com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation.theme.SherpaOnnxSimulateStreamingAsrWearOsTheme
import androidx.wear.compose.material.Text


@Composable
fun HomeScreen() {
    val activity = LocalContext.current as Activity

    var firstTime by remember { mutableStateOf(true) }
    var isStarted by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }

    val onButtonClick: () -> Unit = {
        firstTime = false;
        isStarted = !isStarted

    }

    SherpaOnnxSimulateStreamingAsrWearOsTheme {
        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(MaterialTheme.colors.background),
            contentAlignment = Alignment.Center
        ) {
            Column(
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Spacer(modifier = Modifier.height(16.dp))
                if (firstTime) {
                    ShowMessage()
                } else {
                    ShowResult(result)
                }

                Spacer(modifier = Modifier.height(32.dp))

                Button(
                    onClick = onButtonClick
                ) {
                    if (isStarted) {
                        Text("Stop")
                    } else {
                        Text("Start")
                    }
                }
            }
        }
    }

}
@Composable
fun ShowMessage() {
    val msg = "Real-time\nspeech recognition\nwith\nNext-gen Kaldi"
    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colors.primary,
        text = msg,
    )
}

@Composable
fun ShowResult(result: String) {
    var msg: String = result
    if (msg.length > 10) {
        val n = 5;
        val first = result.take(n);
        val last = result.takeLast(result.length - n)
        msg = "${first}\n${last}"
    }
    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colors.primary,
        text = msg,
    )
}