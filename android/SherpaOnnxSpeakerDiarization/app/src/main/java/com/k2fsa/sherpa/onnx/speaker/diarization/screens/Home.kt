package com.k2fsa.sherpa.onnx.speaker.diarization.screens

import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.documentfile.provider.DocumentFile
import com.k2fsa.sherpa.onnx.speaker.diarization.SpeakerDiarizationObject
import com.k2fsa.sherpa.onnx.speaker.diarization.TAG
import kotlin.concurrent.thread


private var samples: FloatArray? = null

@Composable
fun HomeScreen() {
    val context = LocalContext.current

    var sampleRate: Int
    var filename by remember { mutableStateOf("") }
    var status by remember { mutableStateOf("") }
    var progress by remember { mutableStateOf("") }
    val clipboardManager = LocalClipboardManager.current
    var done by remember { mutableStateOf(false) }
    var fileIsOk by remember { mutableStateOf(false) }
    var started by remember { mutableStateOf(false) }
    var numSpeakers by remember { mutableStateOf(0) }
    var threshold by remember { mutableStateOf(0.5f) }


    val callback = here@{ numProcessedChunks: Int, numTotalChunks: Int, arg: Long ->
        Int
        val percent = 100.0 * numProcessedChunks / numTotalChunks
        progress = "%.2f%%".format(percent)
        Log.i(TAG, progress)
        return@here 0
    }

    val launcher = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) {
        it?.let {
            val documentFile = DocumentFile.fromSingleUri(context, it)
            filename = documentFile?.name ?: ""

            progress = ""
            done = false
            fileIsOk = false

            if (filename.isNotEmpty()) {
                val data = readUri(context, it)
                Log.i(TAG, "sample rate: ${data.sampleRate}")
                Log.i(TAG, "numSamples: ${data.samples?.size ?: 0}")
                if (data.msg != null) {
                    Log.i(TAG, "failed to read $filename")
                    status = data.msg
                } else if (data.sampleRate != SpeakerDiarizationObject.sd.sampleRate()) {
                    status =
                        "Expected sample rate: ${SpeakerDiarizationObject.sd.sampleRate()}. Given wave file with sample rate: ${data.sampleRate}"
                } else {
                    samples = data.samples!!
                    fileIsOk = true
                }
            }
        }
    }

    Column(
        modifier = Modifier.padding(10.dp),
        verticalArrangement = Arrangement.Top,
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {

            Button(onClick = {
                launcher.launch(arrayOf("audio/*"))
            }) {
                Text("Select a .wav file")
            }

            Button(enabled = fileIsOk && !started,
                onClick = {
                    Log.i(TAG, "started")
                    Log.i(TAG, "num samples: ${samples?.size}")
                    started = true
                    progress = ""

                    val config = SpeakerDiarizationObject.sd.config
                    config.clustering.numClusters = numSpeakers
                    config.clustering.threshold = threshold

                    SpeakerDiarizationObject.sd.setConfig(config)

                    thread(true) {
                        done = false
                        status = "Started! Please wait"
                        val segments = SpeakerDiarizationObject.sd.processWithCallback(
                            samples!!,
                            callback = callback,
                        )
                        done = true
                        started = false
                        status = ""
                        for (s in segments) {
                            val start = "%.2f".format(s.start)
                            val end = "%.2f".format(s.end)
                            val speaker = "speaker_%02d".format(s.speaker)
                            status += "$start -- $end $speaker\n"
                            Log.i(TAG, "$start -- $end $speaker")
                        }

                        Log.i(TAG, status)
                    }
                }) {
                Text("Start")
            }
            if (progress.isNotEmpty()) {
                Text(progress, fontSize = 25.sp)
            }
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = numSpeakers.toString(),
                onValueChange = {
                    if (it.isEmpty() || it.isBlank()) {
                        numSpeakers = 0
                    } else {
                        numSpeakers = it.toIntOrNull() ?: 0
                    }
                },
                label = {
                    Text("Number of Speakers")
                },
            )
        }

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceEvenly,
            verticalAlignment = Alignment.CenterVertically
        ) {
            OutlinedTextField(
                value = threshold.toString(),
                onValueChange = {
                    if (it.isEmpty() || it.isBlank()) {
                        threshold = 0.5f
                    } else {
                        threshold = it.toFloatOrNull() ?: 0.5f
                    }
                },
                label = {
                    Text("Clustering threshold")
                },
            )
        }

        if (filename.isNotEmpty()) {
            Text(text = "Selected $filename")
            Spacer(Modifier.size(20.dp))
        }

        if (done) {
            Button(onClick = {
                clipboardManager.setText(AnnotatedString(status))
                progress = "Copied!"
            }) {
                Text("Copy result")
            }
            Spacer(Modifier.size(20.dp))
        }

        if (status.isNotEmpty()) {
            Text(
                status,
                modifier = Modifier.verticalScroll(rememberScrollState()),
            )
        }


    }
}