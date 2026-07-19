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
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream

const val TAG = "sherpa-onnx-pocket-tts"

class MainActivity : ComponentActivity() {
    private var tts: OfflineTts? = null
    private var mediaPlayer: MediaPlayer? = null
    private var generatedWavPath: String = ""

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
                    Toast.makeText(context, "This doesn't look like a valid WAV file", Toast.LENGTH_LONG).show()
                } catch (e: Exception) {
                    Log.e(TAG, "Error checking WAV header", e)
                    Toast.makeText(context, "Failed to read file", Toast.LENGTH_SHORT).show()
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
            Text("PocketTTS Voice Cloning Demo", style = MaterialTheme.typography.h5)

            Button(
                onClick = { 
                    wavPickerLauncher.launch(arrayOf("audio/wav", "audio/x-wav", "audio/wave", "application/octet-stream"))
                }, 
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(if (referenceWavUri != null) "WAV Selected" else "Pick Reference WAV")
            }

            OutlinedTextField(
                value = text,
                onValueChange = { newValue -> text = newValue },
                label = { Text("Text to synthesize") },
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

            Text("Steps: ${steps.toInt()}")
            Slider(value = steps, onValueChange = { steps = it }, valueRange = 1f..50f)

            Text("Temperature: ${String.format("%.2f", temperature)}")
            Slider(value = temperature, onValueChange = { temperature = it }, valueRange = 0.1f..2.0f)

            Text("Speed: ${String.format("%.2f", speed)}")
            Slider(value = speed, onValueChange = { speed = it }, valueRange = 0.5f..2.0f)

            Button(
                onClick = {
                    if (tts == null) {
                        Toast.makeText(context, "TTS not initialized", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    if (referenceWavUri == null) {
                        Toast.makeText(context, "Please pick a reference WAV", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    if (text.isBlank()) {
                        Toast.makeText(context, "Text cannot be empty", Toast.LENGTH_SHORT).show()
                        return@Button
                    }

                    isGenerating = true
                    hasGeneratedAudio = false
                    
                    coroutineScope.launch(Dispatchers.IO) {
                        try {
                            val tempFile = File(context.cacheDir, "ref.wav")
                            context.contentResolver.openInputStream(referenceWavUri!!)?.use { input ->
                                tempFile.outputStream().use { output ->
                                    input.copyTo(output)
                                }
                            }
                            
                            val wave = WaveReader.readWave(tempFile.absolutePath)
                            if (wave.samples.isEmpty()) {
                                withContext(Dispatchers.Main) {
                                    Toast.makeText(context, "Failed to read WAV", Toast.LENGTH_SHORT).show()
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
                                    Toast.makeText(context, "Generation successful!", Toast.LENGTH_SHORT).show()
                                } else {
                                    Toast.makeText(context, "Failed to save generated audio", Toast.LENGTH_SHORT).show()
                                }
                                isGenerating = false
                            }
                        } catch (e: Exception) {
                            Log.e(TAG, "Generation failed", e)
                            withContext(Dispatchers.Main) {
                                Toast.makeText(context, "Generation failed: ${e.message}", Toast.LENGTH_LONG).show()
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
                    Text("Generating...")
                } else {
                    Text("Generate")
                }
            }

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                if (hasGeneratedAudio) {
                    Text("Duration: ${String.format("%.2f", generatedDuration)}s", style = MaterialTheme.typography.body2)
                }

                Button(
                    onClick = {
                        mediaPlayer?.stop()
                        mediaPlayer?.release()
                        mediaPlayer = MediaPlayer.create(context, Uri.fromFile(File(generatedWavPath)))
                        mediaPlayer?.setOnCompletionListener { isPlaying = false }
                        mediaPlayer?.start()
                        isPlaying = true
                    },
                    enabled = hasGeneratedAudio
                ) { Text("Play") }
                
                Button(
                    onClick = {
                        if (isPlaying) {
                            mediaPlayer?.pause()
                            isPlaying = false
                        } else {
                            mediaPlayer?.start()
                            isPlaying = true
                        }
                    },
                    enabled = hasGeneratedAudio
                ) { Text(if (isPlaying) "Pause" else "Resume") }
                
                Button(
                    onClick = {
                        mediaPlayer?.stop()
                        isPlaying = false
                    },
                    enabled = hasGeneratedAudio
                ) { Text("Stop") }
            }

            Button(
                onClick = { saveLauncher.launch("generated.wav") },
                enabled = hasGeneratedAudio,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Save to Device")
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
            Toast.makeText(applicationContext, "Audio saved successfully", Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Log.e(TAG, "Failed to save audio", e)
            Toast.makeText(applicationContext, "Failed to save audio", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        mediaPlayer?.release()
        tts?.release()
    }
}
