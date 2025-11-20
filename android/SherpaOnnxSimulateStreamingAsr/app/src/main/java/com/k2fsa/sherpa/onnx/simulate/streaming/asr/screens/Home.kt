package com.k2fsa.sherpa.onnx.simulate.streaming.asr.screens

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
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.Button
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.stringResource
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import com.k2fsa.sherpa.onnx.simulate.streaming.asr.R
import com.k2fsa.sherpa.onnx.simulate.streaming.asr.SimulateStreamingAsr
import com.k2fsa.sherpa.onnx.simulate.streaming.asr.TAG
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private var audioRecord: AudioRecord? = null

private const val sampleRateInHz = 16000
private var samplesChannel = Channel<FloatArray>(capacity = Channel.UNLIMITED)

@Composable
fun HomeScreen() {
    val context = LocalContext.current
    val clipboardManager = LocalClipboardManager.current

    val activity = LocalContext.current as Activity
    var isStarted by remember { mutableStateOf(false) }
    val resultList: MutableList<String> = remember { mutableStateListOf() }
    val lazyColumnListState = rememberLazyListState()
    val coroutineScope = rememberCoroutineScope()

    var isInitialized by remember { mutableStateOf(false) }

    // we change asrModelType in github actions
    val asrModelType = 15

    LaunchedEffect(Unit) {
        if (asrModelType >= 9000) {
            resultList.add("Using Qnn")
            resultList.add("It takes about 10s for the first run to start")
            resultList.add("Later runs require less than 1 second")
        }

        withContext(Dispatchers.Default) {
            // Call your heavy initialization off the main thread
            SimulateStreamingAsr.initOfflineRecognizer(activity, asrModelType)
            SimulateStreamingAsr.initVad(activity.assets)
        }

        // Back on the Main thread: update UI state
        isInitialized = true
        resultList.clear()
    }

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

                SimulateStreamingAsr.vad.reset()

                CoroutineScope(Dispatchers.IO).launch {
                    Log.i(TAG, "processing samples")
                    val interval = 0.1 // i.e., 100 ms
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
                    val windowSize = 512
                    var isSpeechStarted = false
                    var startTime = System.currentTimeMillis()
                    var lastText = ""
                    var added = false


                    while (isStarted) {
                        for (s in samplesChannel) {
                            if (s.isEmpty()) {
                                break
                            }

                            buffer.addAll(s.toList())
                            while (offset + windowSize < buffer.size) {
                                SimulateStreamingAsr.vad.acceptWaveform(
                                    buffer.subList(
                                        offset,
                                        offset + windowSize
                                    ).toFloatArray()
                                )
                                offset += windowSize
                                if (!isSpeechStarted && SimulateStreamingAsr.vad.isSpeechDetected()) {
                                    isSpeechStarted = true
                                    startTime = System.currentTimeMillis()
                                }
                            }

                            val elapsed = System.currentTimeMillis() - startTime
                            if (isSpeechStarted && elapsed > 200) {
                                // Run ASR every 0.2 seconds == 200 milliseconds
                                // You can change it to some other value
                                val stream = SimulateStreamingAsr.recognizer.createStream()
                                stream.acceptWaveform(
                                    buffer.subList(0, offset).toFloatArray(),
                                    sampleRateInHz
                                )
                                SimulateStreamingAsr.recognizer.decode(stream)
                                val result = SimulateStreamingAsr.recognizer.getResult(stream)
                                stream.release()

                                lastText = result.text

                                if (lastText.isNotBlank()) {
                                    if (!added || resultList.isEmpty()) {
                                        resultList.add(lastText)
                                        added = true
                                    } else {
                                        resultList[resultList.size - 1] = lastText
                                    }

                                    coroutineScope.launch {
                                        lazyColumnListState.animateScrollToItem(resultList.size - 1)
                                    }
                                }

                                startTime = System.currentTimeMillis()
                            }


                            while (!SimulateStreamingAsr.vad.empty()) {
                                val stream = SimulateStreamingAsr.recognizer.createStream()
                                stream.acceptWaveform(
                                    SimulateStreamingAsr.vad.front().samples,
                                    sampleRateInHz
                                )
                                SimulateStreamingAsr.recognizer.decode(stream)
                                val result = SimulateStreamingAsr.recognizer.getResult(stream)
                                stream.release()

                                isSpeechStarted = false
                                SimulateStreamingAsr.vad.pop()

                                buffer = arrayListOf()
                                offset = 0
                                if (lastText.isNotBlank()) {
                                    if (added && resultList.isNotEmpty()) {
                                        resultList[resultList.size - 1] = result.text
                                    } else {
                                        resultList.add(result.text)
                                    }

                                    coroutineScope.launch {
                                        lazyColumnListState.animateScrollToItem(resultList.size - 1)
                                    }
                                    added = false
                                }
                            }
                        }
                    }
                }
            }
        } else {
            audioRecord?.stop()
            audioRecord?.release()
            audioRecord = null
        }
    }



    Box(
        modifier = Modifier.fillMaxSize(),
        contentAlignment = Alignment.TopCenter,
    ) {
        Column(modifier = Modifier) {
            if (!isInitialized) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.Center,
                ) {
                    Text(text = "Initializing... Please wait")


                }
            }

            HomeButtonRow(
                isStarted = isStarted,
                isInitialized = isInitialized,
                onRecordingButtonClick = onRecordingButtonClick,
                onCopyButtonClick = {
                    if (resultList.isNotEmpty()) {
                        val s = resultList.mapIndexed { i, s -> "${i + 1}: $s" }
                            .joinToString(separator = "\n")
                        clipboardManager.setText(AnnotatedString(s))

                        Toast.makeText(
                            context,
                            "Copied to clipboard",
                            Toast.LENGTH_SHORT
                        )
                            .show()
                    } else {
                        Toast.makeText(
                            context,
                            "Nothing to copy",
                            Toast.LENGTH_SHORT
                        )
                            .show()

                    }
                },
                onClearButtonClick = {
                    resultList.clear()
                }
            )

            if (resultList.size > 0) {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .fillMaxHeight(),
                    contentPadding = PaddingValues(16.dp),
                    state = lazyColumnListState
                ) {
                    itemsIndexed(resultList) { index, line ->
                        Text(text = "${index + 1}: $line")
                    }
                }
            }

        }
    }
}

@SuppressLint("UnrememberedMutableState")
@Composable
private fun HomeButtonRow(
    modifier: Modifier = Modifier,
    isStarted: Boolean,
    isInitialized: Boolean,
    onRecordingButtonClick: () -> Unit,
    onCopyButtonClick: () -> Unit,
    onClearButtonClick: () -> Unit,
) {
    Row(
        modifier = modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.Center,
    ) {
        Button(
            onClick = onRecordingButtonClick,
            enabled = isInitialized,
        ) {
            Text(text = stringResource(if (isStarted) R.string.stop else R.string.start))
        }

        Spacer(modifier = Modifier.width(24.dp))

        Button(
            onClick = onCopyButtonClick,
            enabled = isInitialized,
        ) {
            Text(text = stringResource(id = R.string.copy))
        }

        Spacer(modifier = Modifier.width(24.dp))

        Button(
            onClick = onClearButtonClick,
            enabled = isInitialized,
        ) {
            Text(text = stringResource(id = R.string.clear))
        }
    }
}

