package com.k2fsa.sherpa.onnx.fa

import android.content.Context
import android.net.Uri
import com.k2fsa.sherpa.onnx.ForcedAligner
import com.k2fsa.sherpa.onnx.ForcedAlignerConfig
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.Future

/** One completed alignment run, ready for display and playback. */
data class AlignmentRun(
    val words: List<com.k2fsa.sherpa.onnx.AlignmentInterval>,
    val textGrid: String,
    val duration: Float,          // aligned utterance duration in seconds
    val decodeSeconds: Double,
    val elapsedSeconds: Double,   // decode + align
    val samples: FloatArray,      // decoded audio, 16 kHz mono
    val sampleRate: Int,
    val fileName: String,
)

/**
 * Runs forced alignment on a background thread. Publishes status/log lines,
 * progress and the final result through the callbacks.
 *
 * [cancel] abandons the current run: the single compute() call cannot be
 * interrupted, so the worker keeps going until it notices the cancellation.
 * Exactly one of [onResult], [onError] or [onCancelled] is delivered per run,
 * as a single synchronized terminal transition; a cancelled run never
 * delivers a result.
 */
class ForcedAlignmentManager(private val context: Context) {
    var onLog: (String) -> Unit = {}
    var onStatus: (String) -> Unit = {}
    var onProgress: (Float) -> Unit = {}
    var onResult: (AlignmentRun) -> Unit = {}
    var onError: (String) -> Unit = {}
    var onCancelled: () -> Unit = {}

    @Volatile
    var isRunning: Boolean = false
        private set

    // The last completed run; kept here because the Align screen may be
    // disposed (e.g. the Info tab is showing) when a run finishes.
    @Volatile
    var lastResult: AlignmentRun? = null

    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private var future: Future<*>? = null
    private var aligner: ForcedAligner? = null

    // Guards the fields below and the terminal-callback delivery.
    private val lock = Any()
    private var cancelled = false
    private var terminalSent = false
    private var warmingUp = false
    private var pendingRelease = false

    /**
     * Creates the native aligner in the background (model + lexicon are
     * loaded from the APK assets) so the first run is fast. Best-effort:
     * failures are reported by the first real [run] instead.
     */
    fun warmUp() {
        synchronized(lock) {
            if (isRunning || warmingUp || aligner != null) return
            warmingUp = true
        }
        executor.submit {
            try {
                getAligner()
            } catch (e: Throwable) {
                // ignore here; the first run reports the error
            } finally {
                taskFinished(isRun = false)
            }
        }
    }

    /**
     * Runs the alignment. The input is [uri] (a user-picked file) or
     * [cacheFile] (a local file, e.g. the example wav or a microphone recording).
     *
     * @return true if a run was started, false if one is already running or
     *   the input is invalid
     */
    fun run(uri: Uri?, cacheFile: File?, text: String): Boolean {
        val trimmed = text.trim()
        if (trimmed.isEmpty()) {
            onError("Please enter the transcript first")
            return false
        }
        synchronized(lock) {
            if (isRunning) return false
            cancelled = false
            terminalSent = false
            isRunning = true
        }
        onProgress(0f)
        onStatus("Preparing input...")
        future = executor.submit {
            try {
                doRun(uri, cacheFile, trimmed)
            } catch (e: Throwable) {
                // e.g. the model is missing from the APK assets and the
                // native ForcedAligner constructor failed
                deliverError("Error: ${e.message ?: e.javaClass.simpleName}")
            } finally {
                taskFinished(isRun = true)
                onProgress(0f)
            }
        }
        return true
    }

    /**
     * Marks the current run as cancelled. [onCancelled] is delivered when the
     * worker actually stops; until then [isRunning] stays true so a new run
     * cannot start while the old one is still winding down.
     */
    fun cancel() {
        synchronized(lock) {
            if (!isRunning || cancelled) return
            cancelled = true
        }
        onStatus("Cancelled")
        onLog("Cancelled")
    }

    /** Releases the native aligner; call when the activity is destroyed. */
    fun dispose() {
        val needRelease = synchronized(lock) {
            cancelled = true
            if (isRunning || warmingUp) {
                // still using the native aligner; release it when the task
                // finishes (see taskFinished())
                pendingRelease = true
                false
            } else {
                true
            }
        }
        future?.cancel(true)
        executor.shutdown()
        if (needRelease) releaseAligner()
    }

    /** Delivers a result unless this run was cancelled or already finished. */
    private fun deliverResult(run: AlignmentRun) {
        synchronized(lock) {
            if (terminalSent || cancelled) return
            terminalSent = true
            lastResult = run
        }
        onResult(run)
    }

    /** Delivers an error unless this run was cancelled or already finished. */
    private fun deliverError(msg: String) {
        synchronized(lock) {
            if (terminalSent || cancelled) return
            terminalSent = true
        }
        onError(msg)
    }

    private fun isCancelled(): Boolean = synchronized(lock) { cancelled }

    /**
     * Called when a background task (run or warm-up) has finished. Ends the
     * run's terminal state (delivering [onCancelled] if the run was cancelled
     * before delivering anything) and releases the aligner if [dispose] asked
     * for a deferred release while this task was using it.
     */
    private fun taskFinished(isRun: Boolean) {
        var notifyCancelled = false
        val needRelease: Boolean
        synchronized(lock) {
            isRunning = false
            warmingUp = false
            needRelease = pendingRelease
            pendingRelease = false
            if (isRun && cancelled && !terminalSent) {
                terminalSent = true
                notifyCancelled = true
            }
        }
        if (notifyCancelled) onCancelled()
        if (needRelease) releaseAligner()
    }

    private fun releaseAligner() {
        aligner?.release()
        aligner = null
    }

    private fun doRun(uri: Uri?, cacheFile: File?, text: String) {
        // Manager-owned copy of a picked file; deleted when the run ends.
        var ownedInput: File? = null
        try {
            // 1. Resolve the input to a real file (FFmpeg wants a path).
            val inputFile: File
            val fileName: String
            if (cacheFile != null) {
                inputFile = cacheFile
                fileName = cacheFile.name
            } else if (uri != null) {
                onStatus("Copying file...")
                val copied = copyToCache(uri) ?: run {
                    deliverError("Could not read the selected file")
                    return
                }
                ownedInput = copied
                inputFile = copied
                fileName = context.queryDisplayName(uri)
            } else {
                deliverError("No input file")
                return
            }

            // 2. Decode any format to 16 kHz mono float32 PCM.
            onLog("Loaded: $fileName")
            onStatus("Decoding audio...")
            onProgress(0.1f)
            val decodeBegin = System.nanoTime()
            val decoded = AudioDecoder.decode(inputFile, context.cacheDir)
            val decodeSeconds = (System.nanoTime() - decodeBegin) / 1e9
            if (decoded == null) {
                deliverError("Error: Could not decode audio file")
                return
            }
            if (isCancelled()) {
                onStatus("Cancelled")
                return
            }
            onLog(
                "Decoded: ${decoded.duration} s, ${decoded.sampleRate} Hz, " +
                    "${decoded.samples.size} samples in ${decodeSeconds} s"
            )
            onProgress(0.5f)

            // 3. Align.
            onStatus("Aligning...")
            val begin = System.nanoTime()
            val aligner = getAligner()
            val stream = aligner.createStream()
            val result = try {
                stream.acceptWaveform(decoded.samples, decoded.sampleRate)
                aligner.compute(stream, text)
            } finally {
                stream.release()
            }
            if (isCancelled()) {
                onStatus("Cancelled")
                return
            }
            val elapsedSeconds = (System.nanoTime() - begin) / 1e9

            // 4. Collect the word tier.
            val wordsTier = result.tiers.firstOrNull { it.name == "words" }
            val words = wordsTier?.intervals?.filter { it.text.isNotEmpty() }
                ?: emptyList()
            if (words.isEmpty()) {
                deliverError(
                    "Alignment failed (no words). Please check that the " +
                        "transcript matches the audio."
                )
                return
            }

            onProgress(1f)
            onStatus("Done")
            onLog(
                ("Aligned: ${words.size} words | Audio duration: " +
                    "%.3f s | Decode time: %.3f s | Elapsed: %.3f s | " +
                    "RTF = %.3f").format(
                    decoded.duration, decodeSeconds, elapsedSeconds,
                    if (decoded.duration > 0) {
                        elapsedSeconds / decoded.duration
                    } else {
                        0.0
                    },
                )
            )
            deliverResult(
                AlignmentRun(
                    words = words,
                    textGrid = result.textGrid,
                    duration = result.duration,
                    decodeSeconds = decodeSeconds,
                    elapsedSeconds = elapsedSeconds,
                    samples = decoded.samples,
                    sampleRate = decoded.sampleRate,
                    fileName = fileName,
                )
            )
        } finally {
            ownedInput?.delete()
        }
    }

    // Only called from the single-threaded executor, so no locking needed.
    private fun getAligner(): ForcedAligner {
        aligner?.let { return it }
        val model = currentModel
        val a = ForcedAligner(
            assetManager = context.assets,
            config = ForcedAlignerConfig(
                modelDir = model.dir,
                lexicon = "${model.dir}/lexicon.txt",
            ),
        )
        aligner = a
        return a
    }

    private fun copyToCache(uri: Uri): File? {
        return try {
            val suffix = context.queryDisplayName(uri).substringAfterLast('.', "")
            val file = File(context.cacheDir, "fa_input_${System.nanoTime()}.$suffix")
            val input = context.contentResolver.openInputStream(uri) ?: return null
            input.use { src ->
                FileOutputStream(file).use { output -> src.copyTo(output) }
            }
            file
        } catch (e: Exception) {
            null
        }
    }
}
