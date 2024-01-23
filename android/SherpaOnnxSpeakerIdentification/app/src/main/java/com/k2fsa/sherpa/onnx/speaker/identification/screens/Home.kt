package com.k2fsa.sherpa.onnx.speaker.identification.screens

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.onnx.SpeakerRecognition
import com.k2fsa.sherpa.onnx.speaker.identification.R
import com.k2fsa.sherpa.onnx.speaker.identification.TAG
import kotlin.concurrent.thread

private var audioRecord: AudioRecord? = null
private var sampleList: MutableList<FloatArray>? = null

private val clearedResult = "-cleared-"
@Composable
fun HomeScreen() {
    val activity = LocalContext.current as Activity
    var threshold by remember {
        mutableStateOf(0.5F)
    }

    var detectedName by remember {
        mutableStateOf(clearedResult)
    }

    var isStarted by remember { mutableStateOf(false) }
    val onRecordingButtonClick: () -> Unit = {
        isStarted = !isStarted

        if (isStarted) {
            if (ActivityCompat.checkSelfPermission(
                    activity,
                    Manifest.permission.RECORD_AUDIO
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

                sampleList = null
                detectedName = clearedResult

                // recording is started here
                thread(true) {
                    Log.i(TAG, "processing samples")

                    val interval = 0.1 // i.e., 100 ms
                    val bufferSize = (interval * sampleRateInHz).toInt() // in samples
                    val buffer = ShortArray(bufferSize)
                    audioRecord?.let {
                        it.startRecording()

                        while (isStarted) {
                            val ret = audioRecord?.read(buffer, 0, buffer.size)
                            ret?.let { n ->
                                val samples = FloatArray(n) { buffer[it] / 32768.0f }
                                if (sampleList == null) {
                                    sampleList = mutableListOf(samples)
                                } else {
                                    sampleList?.add(samples)
                                }
                            }
                        }
                    }

                    Log.i(TAG, "Home: Recording is stopped. ${sampleList?.count()}")
                }
            }
        } else {
            // recording is stopped here
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null

            sampleList?.let {
                val stream = SpeakerRecognition.extractor.createStream()
                for (samples in it) {
                    stream.acceptWaveform(samples = samples, sampleRate = sampleRateInHz)
                }
                stream.inputFinished()
                if (SpeakerRecognition.extractor.isReady(stream)) {
                    val embedding = SpeakerRecognition.extractor.compute(stream)
                    detectedName = SpeakerRecognition.manager.search(
                        embedding = embedding,
                        threshold = threshold,
                    )
                }
            }
        }
    }

    val onThresholdChange = { newValue: Float ->
        threshold = newValue
    }

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter,
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            HomeThresholdRow(
                threshold = threshold,
                onValueChange = onThresholdChange,
            )
            HomeButtonRow(
                isStarted = isStarted,
                onRecordingButtonClick = onRecordingButtonClick,
                onClearButtonClick = {
                    detectedName = clearedResult
                },
            )

            Spacer(modifier = Modifier.height(48.dp))

            if(detectedName == clearedResult) {
                // do nothing
            } else if (detectedName.length > 0) {
                Text(
                    text = "Speaker: ${detectedName}",
                    style = MaterialTheme.typography.headlineLarge,
                    fontWeight = FontWeight.Bold,
                )
            } else {
                Text(
                    text = "Unknown speaker",
                    style = MaterialTheme.typography.headlineLarge,
                    fontWeight = FontWeight.Bold,
                )
            }
        }
    }
}

@SuppressLint("UnrememberedMutableState")
@Composable
private fun HomeButtonRow(
    modifier: Modifier = Modifier,
    isStarted: Boolean,
    onRecordingButtonClick: () -> Unit,
    onClearButtonClick: () -> Unit,
) {
    val numSpeakers: Int by mutableStateOf(SpeakerRecognition.manager.numSpeakers())
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
    ) {
        Button(
            enabled = numSpeakers > 0,
            onClick = onRecordingButtonClick
        ) {
            Text(text = stringResource(if (isStarted) R.string.stop else R.string.start))
        }

        Spacer(modifier = Modifier.width(24.dp))

        Button(onClick = onClearButtonClick) {
            Text(text = stringResource(id = R.string.clear))
        }
    }
}

@Composable
fun HomeThresholdRow(
    modifier: Modifier = Modifier,
    threshold: Float,
    onValueChange: (Float) -> Unit,
) {
    Column(modifier = Modifier) {
        Text(
            text = "Threshold: " + String.format("%.2f", threshold),
            style = MaterialTheme.typography.headlineMedium,
            fontWeight = FontWeight.Bold,
            modifier = modifier.padding(bottom = 8.dp, top = 8.dp),
        )
        Slider(
            value = threshold,
            onValueChange = onValueChange,
            valueRange = 0.1F..1.0F,
            modifier = modifier.fillMaxWidth(),
        )
    }
}