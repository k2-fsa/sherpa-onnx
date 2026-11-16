package com.k2fsa.sherpa.onnx.fa.screens

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
import android.os.SystemClock
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Checkbox
import androidx.compose.material3.Divider
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.media3.common.MediaItem
import androidx.media3.exoplayer.ExoPlayer
import androidx.media3.ui.PlayerView
import com.k2fsa.sherpa.onnx.AlignmentInterval
import com.k2fsa.sherpa.onnx.fa.AlignmentRun
import com.k2fsa.sherpa.onnx.fa.AudioRecorder
import com.k2fsa.sherpa.onnx.fa.ForcedAlignmentManager
import com.k2fsa.sherpa.onnx.fa.WavEncoder
import com.k2fsa.sherpa.onnx.fa.WordPlayer
import com.k2fsa.sherpa.onnx.fa.currentModel
import com.k2fsa.sherpa.onnx.fa.queryDisplayName
import com.k2fsa.sherpa.onnx.fa.shareTextAsFile
import com.k2fsa.sherpa.onnx.fa.ui.theme.ErrorRed
import com.k2fsa.sherpa.onnx.fa.ui.theme.Muted
import java.io.File
import java.io.FileOutputStream
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.delay

private const val FORMAT_TABLE = "Table"
private const val FORMAT_TIMESTAMPED = "Timestamped"
private const val FORMAT_PLAIN = "Plain"

/** Formats seconds as mm:ss.cc (like tauri-examples/doubao-asr-file). */
private fun formatClock(t: Float): String {
    val m = (t / 60).toInt()
    val s = t - m * 60
    return "%02d:%05.2f".format(Locale.US, m, s)
}

@Composable
fun AlignScreen(manager: ForcedAlignmentManager) {
    val context = LocalContext.current
    val scrollState = rememberScrollState()

    val player = remember { ExoPlayer.Builder(context).build() }

    // rememberSaveable: survive configuration changes (e.g. rotation).
    var transcript by rememberSaveable { mutableStateOf("") }
    var fileName by rememberSaveable { mutableStateOf<String?>(null) }
    var inputUri by rememberSaveable { mutableStateOf<Uri?>(null) }
    // A local cache-file input: the Example wav or a microphone recording.
    var cacheFile by rememberSaveable { mutableStateOf<File?>(null) }

    // The manager survives tab switches and keeps the last completed run;
    // restore it so returning from the Info tab does not lose the result.
    var result by remember { mutableStateOf(manager.lastResult) }
    // the manager outlives this screen (tab switches); a run may already be
    // in progress when the screen is (re)created
    var isProcessing by remember { mutableStateOf(manager.isRunning) }
    var progress by remember { mutableStateOf(0f) }
    var status by remember { mutableStateOf("Idle") }
    var error by remember { mutableStateOf<String?>(null) }
    var logs by remember { mutableStateOf(listOf<String>()) }

    var viewFormat by rememberSaveable { mutableStateOf(FORMAT_TABLE) }
    var searchQuery by rememberSaveable { mutableStateOf("") }
    var hitIndex by remember { mutableStateOf(-1) }
    var subtitlesOn by rememberSaveable { mutableStateOf(true) }
    var playingAll by rememberSaveable { mutableStateOf(false) }

    // Word-level playback state (sample-accurate, see WordPlayer).
    val wordPlayer = remember { WordPlayer() }
    var playingIndex by remember { mutableStateOf(-1) }
    var currentPositionMs by remember { mutableStateOf(0L) }

    // Microphone recording state.
    val recorder = remember { AudioRecorder() }
    var isRecording by remember { mutableStateOf(false) }
    var recordSeconds by remember { mutableStateOf(0) }
    var recordStartMs by remember { mutableStateOf(0L) }

    // Pending word for the save-wav launcher.
    var pendingSaveWord by remember { mutableStateOf<AlignmentInterval?>(null) }

    fun appendLog(msg: String) {
        val stamp = SimpleDateFormat("HH:mm:ss", Locale.US).format(Date())
        logs = (logs + "[$stamp] $msg").takeLast(200) // keep memory bounded
    }

    // ---- manager callbacks ----
    LaunchedEffect(manager) {
        manager.onLog = { appendLog(it) }
        manager.onStatus = { status = it }
        manager.onProgress = { progress = it }
        manager.onError = {
            error = it
            status = "Error"
            isProcessing = false
            appendLog(it)
        }
        manager.onResult = {
            result = it
            isProcessing = false
            hitIndex = -1
        }
        manager.onCancelled = { isProcessing = false }
    }

    // ---- preview-player position polling: subtitle overlay ----
    LaunchedEffect(player) {
        while (true) {
            currentPositionMs = player.currentPosition
            delay(50)
        }
    }

    // ---- recording timer ----
    LaunchedEffect(isRecording) {
        while (isRecording) {
            recordSeconds = ((SystemClock.elapsedRealtime() - recordStartMs) / 1000).toInt()
            delay(200)
        }
    }

    DisposableEffect(player) {
        onDispose { player.release() }
    }
    DisposableEffect(wordPlayer) {
        onDispose { wordPlayer.stop() }
    }
    DisposableEffect(recorder) {
        // do not keep capturing when the screen leaves composition
        onDispose { recorder.stop() }
    }

    // ---- actions ----
    fun loadPlayer(uri: Uri) {
        player.setMediaItem(MediaItem.fromUri(uri))
        player.prepare()
    }

    // Restore the selected media into a fresh player (created after a tab
    // switch or an activity recreation).
    LaunchedEffect(player, inputUri, cacheFile) {
        val uri = inputUri ?: cacheFile?.let { Uri.fromFile(it) }
        if (uri != null && player.mediaItemCount == 0) {
            player.setMediaItem(MediaItem.fromUri(uri))
            player.prepare()
        }
    }

    fun resetResult() {
        wordPlayer.stop()
        manager.lastResult = null
        result = null
        playingIndex = -1
        hitIndex = -1
    }

    val pickFileLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null && !isProcessing) {
            player.pause()
            inputUri = uri
            cacheFile = null
            resetResult()
            error = null
            fileName = context.queryDisplayName(uri)
            loadPlayer(uri)
            status = "Idle"
            appendLog("Loaded: $fileName")
        }
    }

    val saveWavLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("audio/wav")
    ) { uri ->
        val word = pendingSaveWord
        pendingSaveWord = null
        val run = result
        if (uri == null || word == null || run == null) return@rememberLauncherForActivityResult
        try {
            val start = (word.start * run.sampleRate).toInt().coerceIn(0, run.samples.size)
            val end = (word.end * run.sampleRate).toInt().coerceIn(start, run.samples.size)
            val bytes = WavEncoder.encode(run.samples.copyOfRange(start, end), run.sampleRate)
            val out = context.contentResolver.openOutputStream(uri)
                ?: throw IllegalStateException("could not open the output file")
            out.use { it.write(bytes) }
            appendLog("Saved word wav")
        } catch (e: Exception) {
            error = "Could not save wav: $e"
        }
    }

    val saveTextGridLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain")
    ) { uri ->
        val run = result ?: return@rememberLauncherForActivityResult
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val out = context.contentResolver.openOutputStream(uri)
                ?: throw IllegalStateException("could not open the output file")
            out.use {
                it.write(run.textGrid.toByteArray(Charsets.UTF_8))
            }
            appendLog("Saved TextGrid")
        } catch (e: Exception) {
            error = "Could not save TextGrid: $e"
        }
    }

    val saveTimestampedLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain")
    ) { uri ->
        val run = result ?: return@rememberLauncherForActivityResult
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val text = run.words.joinToString("\n") {
                "[${formatClock(it.start)} -> ${formatClock(it.end)}] ${it.text}"
            }
            val out = context.contentResolver.openOutputStream(uri)
                ?: throw IllegalStateException("could not open the output file")
            out.use {
                it.write(text.toByteArray(Charsets.UTF_8))
            }
            appendLog("Saved timestamped txt")
        } catch (e: Exception) {
            error = "Could not save txt: $e"
        }
    }

    val savePlainLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.CreateDocument("text/plain")
    ) { uri ->
        val run = result ?: return@rememberLauncherForActivityResult
        if (uri == null) return@rememberLauncherForActivityResult
        try {
            val text = run.words.joinToString(" ") { it.text }
            val out = context.contentResolver.openOutputStream(uri)
                ?: throw IllegalStateException("could not open the output file")
            out.use {
                it.write(text.toByteArray(Charsets.UTF_8))
            }
            appendLog("Saved plain txt")
        } catch (e: Exception) {
            error = "Could not save txt: $e"
        }
    }

    fun loadExample() {
        if (isProcessing) return
        try {
            val model = currentModel
            val assetPath = "${model.dir}/${model.exampleWav}"
            val file = File(context.cacheDir, model.exampleWav)
            context.assets.open(assetPath).use { input ->
                FileOutputStream(file).use { output -> input.copyTo(output) }
            }
            player.pause()
            inputUri = null
            cacheFile = file
            resetResult()
            error = null
            fileName = "${model.exampleWav} (example)"
            transcript = model.exampleText
            loadPlayer(Uri.fromFile(file))
            status = "Idle"
            appendLog("Loaded example: $assetPath")
        } catch (e: Exception) {
            error = "Could not load the example wav: $e"
        }
    }

    fun runAlignment() {
        if (isProcessing) {
            manager.cancel() // onCancelled() resets isProcessing
            return
        }
        if (isRecording) {
            error = "Stop the recording first"
            return
        }
        error = null
        resetResult()
        logs = listOf()
        isProcessing =
            manager.run(uri = inputUri, cacheFile = cacheFile, text = transcript)
    }

    fun clearAll() {
        player.pause()
        inputUri = null
        cacheFile = null
        fileName = null
        transcript = ""
        logs = listOf()
        error = null
        status = "Idle"
        progress = 0f
        playingAll = false
        resetResult()
    }

    fun playWord(index: Int) {
        val run = result ?: return
        if (index < 0 || index >= run.words.size) return
        if (playingIndex == index) {
            // toggle: stop this word
            wordPlayer.stop()
            playingIndex = -1
            return
        }
        player.pause() // stop the whole-file preview, if any
        playingIndex = index
        val w = run.words[index]
        wordPlayer.play(run.samples, run.sampleRate, w.start, w.end) {
            if (playingIndex != index) return@play // superseded by another word
            if (playingAll && index + 1 < run.words.size) {
                playWord(index + 1)
            } else {
                playingIndex = -1
            }
        }
    }

    fun startRecording() {
        if (isProcessing) return
        if (recorder.start()) {
            isRecording = true
            recordSeconds = 0
            recordStartMs = SystemClock.elapsedRealtime()
            appendLog("Recording started")
        } else {
            error = "Could not start recording (is the microphone in use?)"
        }
    }

    fun stopRecording() {
        isRecording = false
        val samples = recorder.stop()
        if (samples == null || samples.isEmpty()) {
            error = "Recording failed (no audio captured)"
            return
        }
        try {
            val file = File(context.cacheDir, "recording.wav")
            file.writeBytes(WavEncoder.encode(samples, AudioRecorder.SAMPLE_RATE))
            player.pause()
            inputUri = null
            cacheFile = file
            resetResult()
            error = null
            fileName = "recording.wav"
            loadPlayer(Uri.fromFile(file))
            status = "Idle"
            appendLog(
                "Recorded: ${samples.size / AudioRecorder.SAMPLE_RATE} s " +
                    "(${AudioRecorder.SAMPLE_RATE} Hz mono)"
            )
        } catch (e: Exception) {
            error = "Could not save the recording: $e"
        }
    }

    val micPermission = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startRecording()
        } else {
            error = "Microphone permission denied"
        }
    }

    // Capture ended unexpectedly: keep the partial recording but report it.
    LaunchedEffect(recorder) {
        recorder.onError = { msg ->
            appendLog(msg)
            stopRecording()
            error = "$msg (the audio captured so far was kept)"
        }
    }

    // ---- search hits ----
    val words = result?.words ?: emptyList()
    val query = searchQuery.trim()
    val hits = if (query.isEmpty()) emptyList() else words.indices.filter {
        words[it].text.contains(query, ignoreCase = true)
    }

    // ---- UI ----
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(horizontal = 12.dp, vertical = 8.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        // ===================== Card 1: input =====================
        CardSection(title = "Input file") {
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(
                    onClick = { pickFileLauncher.launch(arrayOf("audio/*", "video/*")) },
                    enabled = !isProcessing,
                ) {
                    Text("Select audio or video file…")
                }
                Spacer(Modifier.width(8.dp))
                Text(
                    fileName ?: "No file selected",
                    fontSize = 12.sp,
                    color = Muted,
                    maxLines = 1,
                )
            }
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(onClick = { loadExample() }, enabled = !isProcessing) {
                    Text("Example")
                }
                Spacer(Modifier.width(8.dp))
                Text(
                    "Loads the bundled ${currentModel.exampleWav} and its transcript",
                    fontSize = 12.sp,
                    color = Muted,
                )
            }
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                OutlinedButton(
                    onClick = {
                        if (isRecording) {
                            stopRecording()
                        } else if (
                            ContextCompat.checkSelfPermission(
                                context, Manifest.permission.RECORD_AUDIO
                            ) == PackageManager.PERMISSION_GRANTED
                        ) {
                            startRecording()
                        } else {
                            micPermission.launch(Manifest.permission.RECORD_AUDIO)
                        }
                    },
                    enabled = !isProcessing,
                ) {
                    Text(if (isRecording) "Stop recording" else "Record")
                }
                Spacer(Modifier.width(8.dp))
                if (isRecording) {
                    Text(
                        "Recording… ${recordSeconds}s (tap to stop)",
                        fontSize = 12.sp,
                        color = ErrorRed,
                    )
                } else {
                    Text(
                        "Record from the microphone (16 kHz mono)",
                        fontSize = 12.sp,
                        color = Muted,
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(verticalAlignment = Alignment.CenterVertically) {
                Checkbox(
                    checked = subtitlesOn,
                    onCheckedChange = { subtitlesOn = it },
                )
                Text("Subtitles: ${if (subtitlesOn) "On" else "Off"}", fontSize = 13.sp)
                Spacer(Modifier.width(16.dp))
                Text(
                    "The current word is shown over the player",
                    fontSize = 12.sp,
                    color = Muted,
                )
            }
            if (fileName != null) {
                Spacer(Modifier.height(8.dp))
                Box(
                    Modifier
                        .fillMaxWidth()
                        .height(200.dp)
                ) {
                    AndroidView(
                        factory = { ctx ->
                            PlayerView(ctx).apply {
                                this.player = player
                                useController = true
                            }
                        },
                        modifier = Modifier.fillMaxSize(),
                    )
                    if (subtitlesOn) {
                        val currentWord = words.firstOrNull {
                            currentPositionMs / 1000f >= it.start && currentPositionMs / 1000f < it.end
                        }
                        if (currentWord != null) {
                            Text(
                                currentWord.text,
                                color = Color.White,
                                fontSize = 16.sp,
                                fontWeight = FontWeight.Bold,
                                textAlign = TextAlign.Center,
                                modifier = Modifier
                                    .align(Alignment.BottomCenter)
                                    .padding(8.dp)
                                    .background(
                                        Color.Black.copy(alpha = 0.55f),
                                        RoundedCornerShape(6.dp),
                                    )
                                    .padding(horizontal = 10.dp, vertical = 4.dp),
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Button(
                    onClick = { runAlignment() },
                    enabled = isProcessing ||
                        (fileName != null && transcript.isNotBlank() && !isRecording),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = MaterialTheme.colorScheme.primary,
                    ),
                ) {
                    Text(if (isProcessing) "Stop" else "Align")
                }
                OutlinedButton(
                    onClick = {
                        player.pause()
                        if (isProcessing) manager.cancel()
                        isProcessing = false
                        clearAll()
                    },
                ) {
                    Text("Clear")
                }
            }
            if (isProcessing) {
                Spacer(Modifier.height(8.dp))
                LinearProgressIndicator(
                    progress = progress,
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Spacer(Modifier.height(8.dp))
            Text(status, fontSize = 13.sp, color = Muted)
            error?.let {
                Spacer(Modifier.height(4.dp))
                Text(it, fontSize = 13.sp, color = ErrorRed)
            }
            Spacer(Modifier.height(4.dp))
            Text(
                "Results can be exported from the Result card below " +
                    "(TextGrid / timestamped txt / plain txt). " +
                    "Everything runs on this device.",
                fontSize = 12.sp,
                color = Muted,
            )
        }

        // ===================== Card 2: transcript =====================
        CardSection(title = "Transcript") {
            OutlinedTextField(
                value = transcript,
                onValueChange = { transcript = it },
                modifier = Modifier.fillMaxWidth(),
                minLines = 3,
                placeholder = {
                    Text(
                        "Enter the transcript of the audio. Chinese is " +
                            "tokenized per hanzi, English words are split on " +
                            "whitespace; punctuation is ignored.",
                        fontSize = 13.sp,
                    )
                },
            )
        }

        // ===================== Card 3: result =====================
        CardSection(title = "Result") {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(4.dp),
            ) {
                listOf(FORMAT_TABLE, FORMAT_TIMESTAMPED, FORMAT_PLAIN).forEach { fmt ->
                    TextButton(onClick = {
                        viewFormat = fmt
                        hitIndex = -1
                    }) {
                        Text(
                            fmt,
                            fontSize = 12.sp,
                            fontWeight = if (viewFormat == fmt) FontWeight.Bold else FontWeight.Normal,
                        )
                    }
                }
                Spacer(Modifier.width(4.dp))
                OutlinedTextField(
                    value = searchQuery,
                    onValueChange = {
                        searchQuery = it
                        hitIndex = -1
                    },
                    modifier = Modifier.width(140.dp),
                    placeholder = { Text("Search…", fontSize = 12.sp) },
                    textStyle = TextStyle(fontSize = 12.sp),
                    singleLine = true,
                )
                Text(
                    if (hits.isEmpty()) ""
                    else "${if (hitIndex < 0) "-" else hitIndex + 1}/${hits.size}",
                    fontSize = 11.sp,
                    color = Muted,
                )
                TextButton(
                    onClick = {
                        if (hits.isNotEmpty()) hitIndex = (hitIndex - 1 + hits.size) % hits.size
                    },
                    enabled = hits.isNotEmpty(),
                ) { Text("↑") }
                TextButton(
                    onClick = {
                        if (hits.isNotEmpty()) hitIndex = (hitIndex + 1) % hits.size
                    },
                    enabled = hits.isNotEmpty(),
                ) { Text("↓") }
            }
            Text(
                "Click a row (or ▶) to play just that word — cut sample-" +
                    "accurately from the audio, so it never includes the " +
                    "neighbors. Tap again to stop. Play all plays the words " +
                    "in order.",
                fontSize = 12.sp,
                color = Muted,
            )
            Spacer(Modifier.height(8.dp))

            if (result == null) {
                Text("No result yet.", fontSize = 13.sp, color = Muted)
            } else {
                val run = result!!
                val totalSeconds = run.decodeSeconds + run.elapsedSeconds
                val rtf = if (run.duration > 0) totalSeconds / run.duration else 0.0
                Text(
                    "Words: ${words.size}",
                    fontWeight = FontWeight.Bold,
                    fontSize = 13.sp,
                )
                Spacer(Modifier.height(4.dp))
                Text(
                    "Audio duration: ${run.duration.format3()} s · " +
                        "Decode: ${run.decodeSeconds.format3()} s · " +
                        "Align: ${run.elapsedSeconds.format3()} s · " +
                        "Total: ${totalSeconds.format3()} s · " +
                        "RTF = ${rtf.format3()}",
                    fontSize = 12.sp,
                    color = Muted,
                )
                Spacer(Modifier.height(8.dp))
                Divider()
                when (viewFormat) {
                    FORMAT_TABLE -> {
                        Row(Modifier.fillMaxWidth()) {
                            Text(
                                "Word",
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                modifier = Modifier.weight(1.4f),
                            )
                            Text(
                                "Time (s)",
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                modifier = Modifier.weight(1.2f),
                            )
                            Text(
                                "Audio",
                                fontWeight = FontWeight.Bold,
                                fontSize = 12.sp,
                                modifier = Modifier.weight(0.6f),
                            )
                        }
                        Divider()
                        words.forEachIndexed { i, w ->
                            WordRow(
                                index = i,
                                word = w,
                                isPlaying = playingIndex == i,
                                isHit = hits.getOrNull(hitIndex) == i,
                                query = query,
                                showColumns = true,
                                onClick = { playWord(i) },
                                onPlayWord = { playWord(i) },
                                onSave = { pendingSaveWord = w
                                    saveWavLauncher.launch(
                                        "${run.fileName.baseName()}_word_${i + 1}.wav"
                                    )
                                },
                            )
                        }
                    }
                    FORMAT_TIMESTAMPED -> {
                        words.forEachIndexed { i, w ->
                            WordRow(
                                index = i,
                                word = w,
                                isPlaying = playingIndex == i,
                                isHit = hits.getOrNull(hitIndex) == i,
                                query = query,
                                showColumns = false,
                                onClick = { playWord(i) },
                                onPlayWord = { playWord(i) },
                                onSave = {
                                    pendingSaveWord = w
                                    saveWavLauncher.launch(
                                        "${run.fileName.baseName()}_word_${i + 1}.wav"
                                    )
                                },
                            )
                        }
                    }
                    FORMAT_PLAIN -> {
                        Text(
                            buildAnnotatedString {
                                words.forEachIndexed { i, w ->
                                    val start = length
                                    append(w.text)
                                    if (query.isNotEmpty() && w.text.contains(query, true)) {
                                        addStyle(
                                            SpanStyle(background = Color(0xFFFDE68A)),
                                            start,
                                            length,
                                        )
                                    }
                                    append(" ")
                                    addStyle(
                                        SpanStyle(
                                            color = if (i == playingIndex)
                                                MaterialTheme.colorScheme.primary
                                            else Color.Unspecified
                                        ),
                                        start,
                                        length,
                                    )
                                }
                            },
                            fontSize = 15.sp,
                            lineHeight = 24.sp,
                        )
                    }
                }
                Spacer(Modifier.height(8.dp))
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = {
                        saveTextGridLauncher.launch("${run.fileName.baseName()}.TextGrid")
                    }) { Text("Export .TextGrid", fontSize = 11.sp) }
                    TextButton(onClick = {
                        saveTimestampedLauncher.launch("${run.fileName.baseName()}_timestamped.txt")
                    }) { Text("Export timestamped .txt", fontSize = 11.sp) }
                    TextButton(onClick = {
                        savePlainLauncher.launch("${run.fileName.baseName()}.txt")
                    }) { Text("Export plain .txt", fontSize = 11.sp) }
                }
                Text(
                    "Share sends the result to another app " +
                        "(WeChat, email, ...)",
                    fontSize = 12.sp,
                    color = Muted,
                )
                Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    TextButton(onClick = {
                        // .TextGrid.txt: apps like WeChat reject a bare .TextGrid
                        context.shareTextAsFile(
                            run.textGrid,
                            "${run.fileName.baseName()}.TextGrid.txt",
                        )
                    }) { Text("Share .TextGrid.txt", fontSize = 11.sp) }
                    TextButton(onClick = {
                        // one line per word: "word start -- end"
                        val wordsTxt = run.words.joinToString("\n") {
                            "${it.text} ${it.start.format2()} -- ${it.end.format2()}"
                        }
                        context.shareTextAsFile(
                            wordsTxt,
                            "${run.fileName.baseName()}.words.txt",
                        )
                    }) { Text("Share words .txt", fontSize = 11.sp) }
                }
                Row(
                    horizontalArrangement = Arrangement.spacedBy(4.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Checkbox(
                        checked = playingAll,
                        onCheckedChange = { playingAll = it },
                    )
                    Text("Play all", fontSize = 12.sp)
                    Spacer(Modifier.width(8.dp))
                    TextButton(onClick = {
                        player.pause()
                        resetResult()
                    }) {
                        Text("Clear results", fontSize = 11.sp, color = ErrorRed)
                    }
                }
            }
        }

        // ===================== Card 4: log =====================
        CardSection(title = "Log") {
            logs.forEach {
                Text(it, fontSize = 11.sp, fontFamily = FontFamily.Monospace)
            }
        }
    }
}

private fun Float.format2(): String = "%.2f".format(Locale.US, this)
private fun Float.format3(): String = "%.3f".format(Locale.US, this)
private fun Double.format3(): String = "%.3f".format(Locale.US, this)

private fun String.baseName(): String = this.substringBeforeLast('.')

@Composable
private fun CardSection(title: String, content: @Composable () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surface),
        elevation = CardDefaults.cardElevation(defaultElevation = 0.dp),
    ) {
        Column(Modifier.padding(16.dp)) {
            Text(title, fontWeight = FontWeight.Bold, fontSize = 16.sp)
            Spacer(Modifier.height(12.dp))
            content()
        }
    }
}

@Composable
private fun WordRow(
    index: Int,
    word: AlignmentInterval,
    isPlaying: Boolean,
    isHit: Boolean,
    query: String,
    showColumns: Boolean,
    onClick: () -> Unit,
    onPlayWord: () -> Unit,
    onSave: () -> Unit,
) {
    val highlight = if (isHit) Color(0xFFFDE68A) else
        if (isPlaying) MaterialTheme.colorScheme.primary.copy(alpha = 0.15f) else Color.Transparent
    val timeText = "[${formatClock(word.start)} → ${formatClock(word.end)}]"
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(highlight)
            .border(0.5.dp, MaterialTheme.colorScheme.outline.copy(alpha = 0.4f))
            .clickable(onClick = onClick)
            .padding(vertical = 6.dp, horizontal = 4.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        if (showColumns) {
            Text(
                buildAnnotatedString { appendHighlight(word.text, query) },
                fontSize = 14.sp,
                fontWeight = if (isPlaying) FontWeight.Bold else FontWeight.Normal,
                modifier = Modifier.weight(1.4f),
            )
            Text(timeText, fontSize = 11.sp, color = Muted, modifier = Modifier.weight(1.2f))
        } else {
            Text(timeText, fontSize = 11.sp, color = Muted)
            Spacer(Modifier.width(6.dp))
            Text(
                buildAnnotatedString { appendHighlight(word.text, query) },
                fontSize = 14.sp,
                fontWeight = if (isPlaying) FontWeight.Bold else FontWeight.Normal,
                modifier = Modifier.weight(1f),
            )
        }
        Row(
            modifier = Modifier.weight(0.6f),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            TextButton(onClick = onPlayWord) {
                Text(if (isPlaying) "⏹" else "▶", fontSize = 14.sp)
            }
            TextButton(onClick = onSave) {
                Text("💾", fontSize = 13.sp)
            }
        }
    }
}

private fun androidx.compose.ui.text.AnnotatedString.Builder.appendHighlight(
    text: String,
    query: String,
) {
    if (query.isEmpty()) {
        append(text)
        return
    }
    var i = 0
    while (true) {
        val j = text.indexOf(query, i, ignoreCase = true)
        if (j < 0) {
            append(text.substring(i))
            return
        }
        append(text.substring(i, j))
        withStyle(SpanStyle(background = Color(0xFFFDE68A))) {
            append(text.substring(j, j + query.length))
        }
        i = j + query.length
    }
}
