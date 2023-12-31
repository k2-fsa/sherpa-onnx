@file:OptIn(ExperimentalMaterial3Api::class)

package com.k2fsa.sherpa.onnx.tts.engine

import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.wrapContentHeight
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.k2fsa.sherpa.onnx.tts.engine.ui.theme.SherpaOnnxTtsEngineTheme
import java.io.File

const val TAG = "sherpa-onnx-tts-engine"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        TtsEngine.createTts(this.application)
        setContent {
            SherpaOnnxTtsEngineTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    Scaffold(topBar = {
                        TopAppBar(title = { Text("Next-gen Kaldi: TTS") })
                    }) {
                        Box(modifier = Modifier.padding(it)) {
                            Column {
                                Row {
                                    Text("Speed")
                                    Slider(
                                        value = TtsEngine.speedState.value,
                                        onValueChange = { TtsEngine.speed = it },
                                        valueRange = 0.2F..3.0F
                                    )
                                }
                                var testText by remember { mutableStateOf("") }

                                OutlinedTextField(value = testText,
                                    onValueChange = { testText = it },
                                            label = { Text ("Test text") },
                                    modifier = Modifier.fillMaxWidth().wrapContentHeight().padding(16.dp),
                                    singleLine = false,
                                )

                                val numSpeakers = TtsEngine.tts!!.numSpeakers()
                                if (numSpeakers > 1) {
                                    Row {
                                        Text("Speaker ID: (0-${numSpeakers - 1})")
                                        Slider(
                                            value = TtsEngine.speakerIdState.value.toFloat(),
                                            onValueChange = { TtsEngine.speakerId = it.toInt() },
                                            valueRange = 0.0f..(numSpeakers - 1).toFloat(),
                                            steps = 1
                                        )
                                    }
                                }
                                Row {
                                    Button(
                                        modifier = Modifier.padding(20.dp),
                                        onClick = {
                                            Log.i(TAG, "Clicked, text: ${testText}")
                                            if (testText.isBlank() || testText.isEmpty()) {
                                                Toast.makeText(
                                                    applicationContext,
                                                    "Please input a test sentence",
                                                    Toast.LENGTH_SHORT
                                                ).show()
                                            } else {
                                                val audio = TtsEngine.tts!!.generate(
                                                    text = testText,
                                                    sid = TtsEngine.speakerId,
                                                    speed = TtsEngine.speed,
                                                )

                                                val filename =
                                                    application.filesDir.absolutePath + "/generated.wav"
                                                val ok =
                                                    audio.samples.size > 0 && audio.save(filename)

                                                if (ok) {
                                                    val mediaPlayer = MediaPlayer.create(
                                                        applicationContext,
                                                        Uri.fromFile(File(filename))
                                                    )
                                                    mediaPlayer.start()
                                                } else {
                                                    Log.i(TAG, "Failed to generate or save audio")
                                                }
                                            }
                                        }) {
                                        Text("Test")
                                    }

                                    Button(
                                        modifier = Modifier.padding(20.dp),
                                        onClick = {
                                            TtsEngine.speakerId = 0
                                            TtsEngine.speed = 1.0f
                                            testText = ""
                                        }) {
                                        Text("Reset")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}