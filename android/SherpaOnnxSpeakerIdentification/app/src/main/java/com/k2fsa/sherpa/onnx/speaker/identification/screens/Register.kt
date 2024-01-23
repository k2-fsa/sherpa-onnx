package com.k2fsa.sherpa.onnx.speaker.identification.screens

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.onnx.SpeakerRecognition
import com.k2fsa.sherpa.onnx.speaker.identification.R
import com.k2fsa.sherpa.onnx.speaker.identification.TAG
import kotlin.concurrent.thread

private var audioRecord: AudioRecord? = null

private var sampleList: MutableList<FloatArray>? = null

private var embeddingList: MutableList<FloatArray>? = null

val sampleRateInHz = 16000

@SuppressLint("UnrememberedMutableState")
@Preview
@Composable
fun RegisterScreen(modifier: Modifier = Modifier) {
    val activity = LocalContext.current as Activity

    var firstTime by remember { mutableStateOf(true) }
    if (firstTime) {
        firstTime = false
        // clear states
        embeddingList = null
    }

    val numberAudio: Int by mutableStateOf(embeddingList?.count() ?: 0)

    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter
    ) {
        var speakerName by remember { mutableStateOf("") }
        val onSpeakerNameChange = { newName: String -> speakerName = newName }

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

                        Log.i(TAG, "Recording is stopped. ${sampleList?.count()}")

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
                        stream.acceptWaveform(samples=samples, sampleRate=sampleRateInHz)
                    }
                    stream.inputFinished()
                    if(SpeakerRecognition.extractor.isReady(stream)) {
                        val embedding = SpeakerRecognition.extractor.compute(stream)
                        if(embeddingList == null) {
                            embeddingList = mutableListOf(embedding)
                        } else {
                            embeddingList?.add(embedding)
                        }
                    }
                }
            }
        }

        val onAddButtonClick: () -> Unit = {
            if(speakerName.isEmpty() || speakerName.isBlank()) {
                Toast.makeText(
                    activity,
                    "please input a speaker name",
                    Toast.LENGTH_SHORT
                ).show()
            } else if(SpeakerRecognition.manager.contains(speakerName.trim())) {
                Toast.makeText(
                    activity,
                    "A speaker with $speakerName already exists. Please choose a new name",
                    Toast.LENGTH_SHORT
                ).show()
            } else {
                val ok = SpeakerRecognition.manager.add(speakerName.trim(), embedding = embeddingList!!.toTypedArray())
                if(ok) {
                    Log.i(TAG, "Added ${speakerName.trim()} successfully")
                    Toast.makeText(
                        activity,
                        "Added ${speakerName.trim()}",
                        Toast.LENGTH_SHORT
                    ).show()

                    embeddingList = null
                    sampleList = null
                    speakerName = ""
                    firstTime = true
                } else {
                    Log.i(TAG, "Failed to add ${speakerName.trim()}")
                    Toast.makeText(
                        activity,
                        "Failed to add ${speakerName.trim()}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        }

        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            SpeakerNameRow(speakerName = speakerName, onValueChange = onSpeakerNameChange)
            Text(
                "Number of recordings: ${numberAudio}",
                modifier = modifier.padding(24.dp),
                style = MaterialTheme.typography.headlineMedium,
                fontWeight = FontWeight.Bold,
            )
            RegisterSpeakerButtonRow(
                modifier,
                isStarted = isStarted,
                onRecordingButtonClick = onRecordingButtonClick,
                onAddButtonClick = onAddButtonClick,
            )
        }
    }
}

@Composable
fun SpeakerNameRow(
    modifier: Modifier = Modifier,
    speakerName: String,
    onValueChange: (String) -> Unit
) {
    OutlinedTextField(
        value = speakerName,
        onValueChange = onValueChange,
        label = {
            Text("Please input the speaker name")
        },
        singleLine = true,
        modifier = modifier
            .fillMaxWidth()
            .padding(8.dp)
    )
}

@SuppressLint("UnrememberedMutableState")
@Composable
fun RegisterSpeakerButtonRow(
    modifier: Modifier = Modifier,
    isStarted: Boolean,
    onRecordingButtonClick: () -> Unit,
    onAddButtonClick: () -> Unit,
) {
    val numberAudio: Int by mutableStateOf(embeddingList?.count() ?: 0)
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
    ) {
        Button(onClick = onRecordingButtonClick) {
            Text(text = stringResource(if (isStarted) R.string.stop else R.string.start))
        }

        Spacer(modifier = Modifier.width(24.dp))

        Button(
            enabled = numberAudio > 0,
            onClick = onAddButtonClick,
        ) {
            Text(text = stringResource(id = R.string.add))
        }
    }
}
