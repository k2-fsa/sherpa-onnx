package com.k2fsa.sherpa.onnx

import android.content.Intent
import android.media.MediaPlayer
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.res.stringResource
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

const val TAG = "sherpa-onnx-pocket-tts"

class MainActivity : ComponentActivity() {
    private var tts: OfflineTts? = null
    private var mediaPlayer: MediaPlayer? = null
    private var generatedWavPath: String = ""
    private var generationJob: Job? = null

    private val saveLauncher = registerForActivityResult(ActivityResultContracts.CreateDocument("audio/wav")) { uri ->
        if (uri != null) {
            copyGeneratedWavToUri(uri)
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        
        initTts()
        generatedWavPath = File(filesDir, "generated.wav").absolutePath

        setContent {
            val darkColor = androidx.compose.ui.graphics.Color(0xFF1A1A1A)
            val customColors = lightColors(
                background = androidx.compose.ui.graphics.Color.White,
                surface = androidx.compose.ui.graphics.Color.White,
                onBackground = darkColor,
                onSurface = darkColor,
                primary = androidx.compose.ui.graphics.Color(0xFF6200EE)
            )
            MaterialTheme(colors = customColors) {
                Surface(
                    color = MaterialTheme.colors.background,
                    contentColor = MaterialTheme.colors.onBackground,
                    modifier = Modifier.fillMaxSize()
                ) {
                    PocketTtsApp()
                }
            }
        }
    }

    private fun initTts() {
        val modelDir = "sherpa-onnx-pocket-tts-int8-2026-01-26"
        try {
            val config = OfflineTtsConfig(
                model = OfflineTtsModelConfig(
                    pocket = OfflineTtsPocketModelConfig(
                        lmFlow = "$modelDir/lm_flow.int8.onnx",
                        lmMain = "$modelDir/lm_main.int8.onnx",
                        encoder = "$modelDir/encoder.onnx",
                        decoder = "$modelDir/decoder.int8.onnx",
                        textConditioner = "$modelDir/text_conditioner.onnx",
                        vocabJson = "$modelDir/vocab.json",
                        tokenScoresJson = "$modelDir/token_scores.json"
                    ),
                    numThreads = 2,
                    debug = true
                )
            )
            tts = OfflineTts(assetManager = assets, config = config)
            Log.i(TAG, "TTS initialized successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize TTS", e)
        }
    }

    @Composable
    fun PocketTtsApp() {
        var text by remember { mutableStateOf("Today as always, men fall into two groups: slaves and free men.") }
        var referenceWavUri by remember { mutableStateOf<Uri?>(null) }
        var steps by remember { mutableStateOf(5f) }
        var temperature by remember { mutableStateOf(0.7f) }
        var speed by remember { mutableStateOf(1.0f) }
        var isGenerating by remember { mutableStateOf(false) }
        var isPlaying by remember { mutableStateOf(false) }
        var hasGeneratedAudio by remember { mutableStateOf(false) }
        var generatedDuration by remember { mutableStateOf(0f) }

        val context = LocalContext.current
        val coroutineScope = rememberCoroutineScope()

        val wavPickerLauncher = rememberLauncherForActivityResult(
            contract = ActivityResultContracts.OpenDocument()
        ) { uri: Uri? ->
            if (uri != null) {
                try {
                    referenceWavUri?.let { oldUri ->
                        context.contentResolver.releasePersistableUriPermission(oldUri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    }
                    context.contentResolver.takePersistableUriPermission(uri, Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    context.contentResolver.openInputStream(uri)?.use { inputStream ->
                        val buffer = ByteArray(12)
                        val bytesRead = inputStream.read(buffer)
                        if (bytesRead == 12) {
                            val riff = String(buffer, 0, 4)
                            val wave = String(buffer, 8, 4)
                            if (riff == "RIFF" && wave == "WAVE") {
                                referenceWavUri = uri
                                return@rememberLauncherForActivityResult
                            }
                        }
                    }
                    Toast.makeText(context, context.getString(R.string.err_invalid_wav), Toast.LENGTH_LONG).show()
                } catch (e: Exception) {
                    Log.e(TAG, "Error checking WAV header", e)
                    Toast.makeText(context, context.getString(R.string.err_read_file), Toast.LENGTH_SHORT).show()
                }
            }
            referenceWavUri = null
        }

        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(16.dp)
                .verticalScroll(rememberScrollState()),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text(stringResource(R.string.title_main), style = MaterialTheme.typography.h5)

            Button(
                onClick = { 
                    wavPickerLauncher.launch(arrayOf("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"))
                }, 
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (referenceWavUri != null) stringResource(R.string.btn_wav_selected) else stringResource(R.string.btn_pick_wav))
            }

            OutlinedTextField(
                value = text,
                onValueChange = { newValue -> text = newValue },
                label = { Text(stringResource(R.string.label_text_input)) },
                modifier = Modifier.fillMaxWidth(),
                maxLines = 10,
                colors = TextFieldDefaults.outlinedTextFieldColors(
                    textColor = MaterialTheme.colors.onBackground,
                    unfocusedLabelColor = MaterialTheme.colors.onBackground,
                    focusedLabelColor = MaterialTheme.colors.primary,
                    unfocusedBorderColor = androidx.compose.ui.graphics.Color.Gray,
                    focusedBorderColor = MaterialTheme.colors.primary,
                    cursorColor = MaterialTheme.colors.primary
                )
            )

            Text("${stringResource(R.string.label_steps)} ${steps.toInt()}")
            Slider(value = steps, onValueChange = { steps = it }, valueRange = 1f..50f)

            Text("${stringResource(R.string.label_temperature)} ${String.format(Locale.US, "%.2f", temperature)}")
            Slider(value = temperature, onValueChange = { temperature = it }, valueRange = 0.1f..2.0f)

            Text("${stringResource(R.string.label_speed)} ${String.format(Locale.US, "%.2f", speed)}")
            Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.5f..2.0f)

            Button(
                onClick = {
                    if (tts == null) {
                        Toast.makeText(context, context.getString(R.string.err_tts_not_init), Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    val refUri = referenceWavUri
                    if (refUri == null) {
                        Toast.makeText(context, context.getString(R.string.err_pick_wav), Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    if (text.isBlank()) {
                        Toast.makeText(context, context.getString(R.string.err_empty_text), Toast.LENGTH_SHORT).show()
                        return@Button
                    }

                    isGenerating = true
                    hasGeneratedAudio = false
                    
                    mediaPlayer?.stop()
                    mediaPlayer?.release()
                    mediaPlayer = null
                    isPlaying = false
                    
                    generationJob = coroutineScope.launch(Dispatchers.IO) {
                        try {
                            val tempFile = File(context.cacheDir, "ref.wav")
                            context.contentResolver.openInputStream(refUri)?.use { input ->
                                tempFile.outputStream().use { output ->
                                    input.copyTo(output)
                                }
                            }
                            
                            val wave = WaveReader.readWave(tempFile.absolutePath)
                            if (wave.samples.isEmpty()) {
                                withContext(Dispatchers.Main) {
                                    Toast.makeText(context, context.getString(R.string.err_read_wav), Toast.LENGTH_SHORT).show()
                                    isGenerating = false
                                }
                                return@launch
                            }

                            val extra = mapOf(
                                "temperature" to temperature.toString(),
                            )
                            val genConfig = GenerationConfig(
                                referenceAudio = wave.samples,
                                referenceSampleRate = wave.sampleRate,
                                numSteps = steps.toInt(),
                                speed = speed,
                                extra = extra
                            )
                            
                            val audio = tts!!.generateWithConfig(text, genConfig)
                            val success = audio.save(generatedWavPath)
                            val duration = audio.samples.size.toFloat() / audio.sampleRate
                            
                            withContext(Dispatchers.Main) {
                                if (success) {
                                    hasGeneratedAudio = true
                                    generatedDuration = duration
                                    Toast.makeText(context, context.getString(R.string.msg_gen_success), Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(context, context.getString(R.string.err_save_audio), Toast.LENGTH_SHORT).show()
                                }
                                isGenerating = false
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Generation failed", e)
                            withContext(Dispatchers.Main) {
                                Toast.makeText(context, "${context.getString(R.string.err_gen_failed)} ${e.message}", Toast.LENGTH_LONG).show()
                                isGenerating = false
                            }
                        }
                    }
                },
                enabled = !isGenerating,
                modifier = Modifier.fillMaxWidth()
            ) {
                if (isGenerating) {
                    CircularProgressIndicator(modifier = Modifier.size(24.dp), color = MaterialTheme.colors.onPrimary)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(stringResource(R.string.btn_generating))
                } else {
                    Text(stringResource(R.string.btn_generate))
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (hasGeneratedAudio) {
                    Text("${stringResource(R.string.label_duration)} ${String.format(Locale.US, "%.2f", generatedDuration)}s", style = MaterialTheme.typography.body2)
                }

                Button(
                    onClick = {
                        mediaPlayer?.stop()
                        mediaPlayer?.release()
                        val mp = MediaPlayer.create(context, Uri.fromFile(File(generatedWavPath)))
                        if (mp != null) {
                            mediaPlayer = mp
                            mp.setOnCompletionListener { isPlaying = false }
                            mp.start()
                            isPlaying = true
                        } else {
                            mediaPlayer = null
                            isPlaying = false
                            Toast.makeText(context, context.getString(R.string.err_play_audio), Toast.LENGTH_SHORT).show()
                        }
                    },
                    enabled = hasGeneratedAudio
                ) { Text(stringResource(R.string.btn_play)) }
                
                Button(
                    onClick = {
                        val mp = mediaPlayer
                        if (mp != null) {
                            if (isPlaying) {
                                mp.pause()
                                isPlaying = false
                            } else {
                                try {
                                    mp.start()
                                    isPlaying = true
                                } catch (e: IllegalStateException) {
                                    Log.e(TAG, "Failed to start MediaPlayer", e)
                                }
                            }
                        }
                    },
                    enabled = hasGeneratedAudio
                ) { Text(if (isPlaying) stringResource(R.string.btn_pause) else stringResource(R.string.btn_resume)) }
                
                Button(
                    onClick = {
                        mediaPlayer?.pause()
                        mediaPlayer?.seekTo(0)
                        isPlaying = false
                    },
                    enabled = hasGeneratedAudio
                ) { Text(stringResource(R.string.btn_stop)) }
            }

            Button(
                onClick = { saveLauncher.launch("generated.wav") },
                enabled = hasGeneratedAudio,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(stringResource(R.string.btn_save))
            }
        }
    }

    private fun copyGeneratedWavToUri(destUri: Uri) {
        try {
            val srcFile = File(generatedWavPath)
            contentResolver.openOutputStream(destUri)?.use { output ->
                srcFile.inputStream().use { input ->
                    input.copyTo(output)
                }
            }
            Toast.makeText(applicationContext, getString(R.string.msg_audio_saved), Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save audio", e)
            Toast.makeText(applicationContext, getString(R.string.err_save_audio), Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mediaPlayer?.release()
        runBlocking {
            generationJob?.cancelAndJoin()
        }
        tts?.release()
    }
}
