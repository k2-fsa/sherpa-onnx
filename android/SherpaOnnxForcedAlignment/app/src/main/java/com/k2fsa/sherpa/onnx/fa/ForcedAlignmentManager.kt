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
import java.util.concurrent.atomic.AtomicBoolean

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
 * progress and the final result through the callbacks; [cancel] abandons the
 * current run (the single compute() call is not interruptible, so a cancelled
 * run's result is dropped).
 */
class ForcedAlignmentManager(private val context: Context) {
    var onLog: (String) -> Unit = {}
    var onStatus: (String) -> Unit = {}
    var onProgress: (Float) -> Unit = {}
    var onResult: (AlignmentRun) -> Unit = {}
    var onError: (String) -> Unit = {}
    var onCancelled: () -> Unit = {}

    var isRunning: Boolean = false
        private set

    private val executor: ExecutorService = Executors.newSingleThreadExecutor()
    private val cancelled = AtomicBoolean(false)
    private var future: Future<*>? = null
    private var aligner: ForcedAligner? = null
    private var pendingRelease = false

    @Volatile
    private var warmingUp = false

    /**
     * Creates the native aligner in the background (model + lexicon are
     * loaded from the APK assets) so the first run is fast. Best-effort:
     * failures are reported by the first real [run] instead.
     */
    fun warmUp() {
        synchronized(this) {
            if (isRunning || warmingUp || aligner != null) return
            warmingUp = true
        }
        executor.submit {
            try {
                getAligner()
            } catch (e: Throwable) {
                // ignore here; the first run reports the error
            } finally {
                warmingUp = false
                if (pendingRelease) {
                    releaseAligner()
                }
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
        if (isRunning) return false
        val trimmed = text.trim()
        if (trimmed.isEmpty()) {
            onError("Please enter the transcript first")
            return false
        }
        cancelled.set(false)
        isRunning = true
        onProgress(0f)
        onStatus("Preparing input...")
        future = executor.submit {
            try {
                doRun(uri, cacheFile, trimmed)
            } catch (e: Throwable) {
                // e.g. the model is missing from the APK assets and the
                // native ForcedAligner constructor failed
                onError("Error: ${e.message ?: e.javaClass.simpleName}")
            } finally {
                isRunning = false
                onProgress(0f)
                if (pendingRelease) {
                    releaseAligner()
                }
            }
        }
        return true
    }

    fun cancel() {
        if (!isRunning) return
        cancelled.set(true)
        onStatus("Cancelled")
        onLog("Cancelled")
        onCancelled()
    }

    /** Releases the native aligner; call when the activity is destroyed. */
    fun dispose() {
        cancelled.set(true)
        future?.cancel(true)
        executor.shutdown()
        if (isRunning || warmingUp) {
            // still using the native aligner; release it when the task
            // finishes (see the finally blocks of run()/warmUp())
            pendingRelease = true
        } else {
            releaseAligner()
        }
    }

    private fun releaseAligner() {
        aligner?.release()
        aligner = null
    }

    private fun doRun(uri: Uri?, cacheFile: File?, text: String) {
        // 1. Resolve the input to a real file (FFmpeg wants a path).
        val inputFile: File
        val fileName: String
        if (cacheFile != null) {
            inputFile = cacheFile
            fileName = cacheFile.name
        } else if (uri != null) {
            onStatus("Copying file...")
            val copied = copyToCache(uri) ?: run {
                onError("Could not read the selected file")
                return
            }
            inputFile = copied
            fileName = context.queryDisplayName(uri)
        } else {
            onError("No input file")
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
            onError("Error: Could not decode audio file")
            return
        }
        if (cancelled.get()) {
            onStatus("Cancelled")
            onCancelled()
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
        if (cancelled.get()) {
            onStatus("Cancelled")
            onCancelled()
            return
        }
        val elapsedSeconds = (System.nanoTime() - begin) / 1e9

        // 4. Collect the word tier.
        val wordsTier = result.tiers.firstOrNull { it.name == "words" }
        val words = wordsTier?.intervals?.filter { it.text.isNotEmpty() } ?: emptyList()
        if (words.isEmpty()) {
            onError(
                "Alignment failed (no words). Please check that the " +
                    "transcript matches the audio."
            )
            return
        }

        onProgress(1f)
        onStatus("Done")
        onLog(
            "Aligned: ${words.size} words | Audio duration: " +
                "%.3f s | Decode time: %.3f s | Elapsed: %.3f s | RTF = %.3f"
                    .format(
                        decoded.duration, decodeSeconds, elapsedSeconds,
                        if (decoded.duration > 0) elapsedSeconds / decoded.duration else 0.0,
                    )
        )
        onResult(
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
    }

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
