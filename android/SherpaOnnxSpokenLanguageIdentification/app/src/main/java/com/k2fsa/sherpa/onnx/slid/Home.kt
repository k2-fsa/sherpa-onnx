@file:OptIn(ExperimentalMaterial3Api::class)

package com.k2fsa.sherpa.onnx.slid

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import kotlin.concurrent.thread

@Composable
fun Home() {
    Scaffold(
        topBar = {
            CenterAlignedTopAppBar(
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.primary,
                ),
                title = {
                    Text(
                        "Next-gen Kaldi: Spoken language identification",
                        fontWeight = FontWeight.Bold,
                        fontSize = 13.sp,
                    )
                },
            )
        },
        content = {
            MyApp(it)
        },
    )
}

private var audioRecord: AudioRecord? = null
private const val sampleRateInHz = 16000

@Composable
fun MyApp(padding: PaddingValues) {
    val activity = LocalContext.current as Activity
    var isStarted by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }

    val onButtonClick: () -> Unit = {
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
                    val samples = flatten(sampleList)
                    val stream = Slid.slid.createStream()
                    stream.acceptWaveform(samples, sampleRateInHz)
                    val lang = Slid.slid.compute(stream)

                    result = Slid.localeMap[lang] ?: lang

                    stream.release()
                }
            }
        }
    }

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter
    ) {
        Column(
            Modifier.padding(padding),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(modifier = Modifier.height(16.dp))
            Button(onClick = onButtonClick) {
                if (isStarted) {
                    Text("Stop")
                } else {
                    Text("Start")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            if (result.isNotEmpty() && result.isNotBlank()) {
                Text("Detected language: $result")
            }
        }
    }
}

fun flatten(sampleList: ArrayList<FloatArray>): FloatArray {
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