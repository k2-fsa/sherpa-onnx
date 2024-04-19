@file:OptIn(ExperimentalMaterial3Api::class, ExperimentalFoundationApi::class)

package com.k2fsa.sherpa.onnx.audio.tagging

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.CenterAlignedTopAppBar
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.onnx.AudioEvent
import com.k2fsa.sherpa.onnx.Tagger
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
                        "Next-gen Kaldi: Audio tagging",
                        fontWeight = FontWeight.Bold,
                        fontSize = 15.sp,
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
private val sampleRateInHz = 16000

@Composable
fun MyApp(padding: PaddingValues) {
    val activity = LocalContext.current as Activity
    var threshold by remember { mutableStateOf<Float>(0.6F) }
    var isStarted by remember { mutableStateOf(false) }
    val result = remember { mutableStateListOf<AudioEvent>() }


    val onButtonClick: () -> Unit = {
        isStarted = !isStarted
        if (isStarted) {
            result.clear()
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
                    for (e in events) {
                        if (e.prob > threshold) {
                            result.add(e)
                        }

                    }

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
            Text("Threshold " + String.format("%.1f", threshold))
            Slider(
                value = threshold,
                onValueChange = { threshold = it },
                valueRange = 0.1F..1.0F,
                modifier = Modifier.fillMaxWidth()
            )

            Button(onClick = onButtonClick) {
                if (isStarted) {
                    Text("Stop")
                } else {
                    Text("Start")
                }
            }

            Spacer(modifier = Modifier.height(16.dp))
            LazyColumn(modifier = Modifier.fillMaxSize()) {
                if (!result.isEmpty()) {

                    item {
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly
                        ) {
                            Text(
                                text = "Event name",
                            )
                            Text(
                                text = "Probability",
                            )
                        }
                    }
                }

                items(result) { event: AudioEvent ->
                    ViewRow(event = event)
                }
            }
        }
    }
}

@Composable
fun ShowResult(result: String) {
    Text(
        modifier = Modifier.fillMaxWidth(),
        textAlign = TextAlign.Center,
        color = MaterialTheme.colorScheme.primary,
        text = result,
    )
}

@Composable
fun ViewRow(
    modifier: Modifier = Modifier,
    event: AudioEvent
) {
    Surface(
        modifier = modifier
            .fillMaxWidth()
            .padding(8.dp),
        color = MaterialTheme.colorScheme.inversePrimary,
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