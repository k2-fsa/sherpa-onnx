package com.k2fsa.sherpa.onnx.audio.tagging.wear.os.presentation

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.Slider
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
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import com.k2fsa.sherpa.onnx.AudioEvent
import com.k2fsa.sherpa.onnx.Tagger
import com.k2fsa.sherpa.onnx.audio.tagging.wear.os.presentation.theme.SherpaOnnxAudioTaggingWearOsTheme
import kotlin.concurrent.thread

private var audioRecord: AudioRecord? = null
private val sampleRateInHz = 16000

@Composable
fun HomeScreen() {
    val activity = LocalContext.current as Activity
    var threshold by remember { mutableStateOf<Float>(0.6F) }
    var firstTime by remember { mutableStateOf(true) }
    var isStarted by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }
    val onButtonClick: () -> Unit = {
        firstTime = false

        isStarted = !isStarted
        if (isStarted) {
            result = ""
            if (ActivityCompat.checkSelfPermission(
                    activity,
                    Manifest.permission.RECORD_AUDIO
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                Log.i(TAG, "Recording is not allowed")
            } else {
                val audioSource = MediaRecorder.AudioSource.MIC
                val channelConfig = AudioFormat.CHANNEL_IN_MONO
                val audioFormat = AudioFormat.ENCODING_PCM_16BIT
                val numBytes =
                    AudioRecord.getMinBufferSize(sampleRateInHz, channelConfig, audioFormat)

                audioRecord = AudioRecord(
                    audioSource,
                    sampleRateInHz,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    numBytes * 2 // a sample has two bytes as we are using 16-bit PCM
                )

                thread(true) {
                    Log.i(TAG, "processing samples")
                    val interval = 0.1 // i.e., 100 ms
                    val bufferSize = (interval * sampleRateInHz).toInt() // in samples
                    val buffer = ShortArray(bufferSize)
                    val sampleList = ArrayList<FloatArray>()
                    audioRecord?.let {
                        it.startRecording()
                        while (isStarted) {
                            val ret = it.read(buffer, 0, buffer.size)
                            ret.let { n ->
                                val samples = FloatArray(n) { buffer[it] / 32768.0f }
                                sampleList.add(samples)
                            }
                        }
                    }
                    Log.i(TAG, "Stop recording")
                    Log.i(TAG, "Start recognition")
                    val samples = Flatten(sampleList)
                    val stream = Tagger.tagger.createStream()
                    stream.acceptWaveform(samples, sampleRateInHz)
                    val events = Tagger.tagger.compute(stream)
                    stream.release()

                    var str: String = ""
                    for (e in events) {
                        if (e.prob > threshold) {
                            str += "%s (%.2f)\n".format(e.name, e.prob)
                        }
                    }
                    result = str
                }
            }
        }
    }


    SherpaOnnxAudioTaggingWearOsTheme {
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
                }

                Spacer(modifier = Modifier.height(16.dp))
                Text(
                    result,
                    fontSize = 12.sp,
                )

                Text(
                    "Threshold " + String.format("%.1f", threshold),
                    fontSize = 12.sp
                )
                Slider(
                    value = threshold,
                    onValueChange = { threshold = it },
                    valueRange = 0.1F..1.0F,
                    modifier = Modifier.fillMaxWidth()
                )
                Button(
                    onClick = onButtonClick,
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
    val msg = "Audio tagging\nwith\nNext-gen Kaldi"
    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colors.primary,
        text = msg,
    )
}

@Composable
fun ViewRow(
    modifier: Modifier = Modifier,
    event: AudioEvent
) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.Center,
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text = event.name,
            modifier = modifier.weight(1.0F),
        )
        Text(
            text = "%.2f".format(event.prob),
            modifier = modifier.weight(1.0F),
        )
    }

}


fun Flatten(sampleList: ArrayList<FloatArray>): FloatArray {
    var totalSamples = 0
    for (a in sampleList) {
        totalSamples += a.size
    }
    var i = 0
    val samples = FloatArray(totalSamples)
    for (a in sampleList) {
        for (s in a) {
            samples[i] = s
            i += 1
        }
    }
    Log.i(TAG, "$i, $totalSamples")

    return samples
}