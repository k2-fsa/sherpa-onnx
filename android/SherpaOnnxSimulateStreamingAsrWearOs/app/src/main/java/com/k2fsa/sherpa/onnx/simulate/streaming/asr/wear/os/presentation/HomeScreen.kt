package com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation

import android.Manifest
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
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
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import androidx.wear.compose.material.Button
import androidx.wear.compose.material.MaterialTheme
import androidx.wear.compose.material.Text
import com.k2fsa.sherpa.onnx.simulate.streaming.asr.wear.os.presentation.theme.SherpaOnnxSimulateStreamingAsrWearOsTheme
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch


private var audioRecord: AudioRecord? = null

private const val sampleRateInHz = 16000
private var samplesChannel = Channel<FloatArray>(capacity = Channel.UNLIMITED)

@Composable
fun HomeScreen() {
    val activity = LocalContext.current as Activity

    var firstTime by remember { mutableStateOf(true) }
    var isStarted by remember { mutableStateOf(false) }
    var result by remember { mutableStateOf("") }

    val coroutineScope = rememberCoroutineScope()

    val onButtonClick: () -> Unit = {
        firstTime = false
        isStarted = !isStarted


        if (isStarted) {
            if (ActivityCompat.checkSelfPermission(
                    activity, Manifest.permission.RECORD_AUDIO
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                Log.i(TAG, "Recording is not allowed")
            } else {
                // recording is allowed
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

                SimulateStreamingAsr.vad.reset()

                result = "Started! Please speak"

                CoroutineScope(Dispatchers.IO).launch {
                    Log.i(TAG, "processing samples")
                    val interval = 0.2 // i.e., 200 ms
                    val bufferSize = (interval * sampleRateInHz).toInt() // in samples
                    val buffer = ShortArray(bufferSize)

                    audioRecord?.let { it ->
                        it.startRecording()

                        while (isStarted) {
                            val ret = audioRecord?.read(buffer, 0, buffer.size)
                            ret?.let { n ->
                                val samples = FloatArray(n) { buffer[it] / 32768.0f }
                                samplesChannel.send(samples)
                            }
                        }
                        val samples = FloatArray(0)
                        samplesChannel.send(samples)
                    }
                }

                CoroutineScope(Dispatchers.Default).launch {
                    var buffer = arrayListOf<Float>()
                    var offset = 0
                    val windowSize = 512 // change it for ten-vad

                    while (isStarted) {
                        for (s in samplesChannel) {
                            if (s.isEmpty()) {
                                break
                            }

                            buffer.addAll(s.toList())
                            while (offset + windowSize < buffer.size) {
                                SimulateStreamingAsr.vad.acceptWaveform(
                                    buffer.subList(
                                        offset, offset + windowSize
                                    ).toFloatArray()
                                )

                                offset += windowSize
                            }

                            while (!SimulateStreamingAsr.vad.empty()) {
                                val duration = SimulateStreamingAsr.vad.front().samples.count().toFloat() / 16000

                                val s0 = System.currentTimeMillis()
                                val stream = SimulateStreamingAsr.recognizer.createStream()
                                stream.acceptWaveform(
                                    SimulateStreamingAsr.vad.front().samples,
                                    sampleRateInHz
                                )
                                SimulateStreamingAsr.recognizer.decode(stream)

                                val s1 = System.currentTimeMillis()
                                val diff = (s1 - s0).toFloat() / 1000
                                val rtf = diff / duration
                                Log.i(TAG, "rtf: ${rtf}, elapsed: ${diff}, duration: ${duration}")
                                val r = SimulateStreamingAsr.recognizer.getResult(stream)
                                stream.release()

                                Log.i(TAG, "result: ${r.text}")

                                coroutineScope.launch {
                                    result = r.text
                                }

                                SimulateStreamingAsr.vad.pop()
                                buffer = arrayListOf()
                                offset = 0
                            }
                        }
                    }
                }
            }
        } else {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            result = "Click Start and speak"
        }
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
        val n = 5
        val first = result.take(n)
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