package com.k2fsa.sherpa.onnx.fa

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import java.io.ByteArrayOutputStream

/**
 * Records from the microphone as 16 kHz mono Float32 PCM (the same format
 * [AudioDecoder] produces and the aligner wants).
 *
 * Usage: [start], then [stop] to get the recorded samples. If capture ends
 * unexpectedly (e.g. a read error), [onError] is called and the samples
 * captured so far stay available to [stop].
 */
class AudioRecorder {
    companion object {
        const val SAMPLE_RATE = 16000
    }

    /** Called from the recording thread when capture terminates unexpectedly. */
    var onError: (String) -> Unit = {}

    private var record: AudioRecord? = null
    private var thread: Thread? = null

    // The buffer of the current session; only touched by the session's worker
    // thread and by stop() after that worker has exited.
    private var sessionSamples: ByteArrayOutputStream? = null

    @Volatile
    private var running = false

    val isRunning: Boolean
        get() = running

    /**
     * Starts recording; returns false if the microphone is not available.
     *
     * The caller must hold RECORD_AUDIO — the UI checks the permission (and
     * requests it if needed) before calling this.
     */
    @SuppressLint("MissingPermission") // permission is checked by the caller
    @Synchronized
    fun start(): Boolean {
        if (running) return false
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val rec = try {
            AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT,
                maxOf(minBuf * 2, 8192),
            )
        } catch (e: SecurityException) {
            // no RECORD_AUDIO permission
            return false
        }
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return false
        }
        val samples = ByteArrayOutputStream()
        sessionSamples = samples
        record = rec
        running = true
        thread = Thread {
            val buf = ShortArray(2048)
            try {
                rec.startRecording()
                while (running) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n < 0) {
                        // recording error; keep the samples captured so far
                        running = false
                        onError("Recording failed (AudioRecord.read error $n)")
                        return@Thread
                    }
                    if (n == 0) continue
                    synchronized(samples) {
                        for (i in 0 until n) {
                            val v = buf[i].toInt()
                            samples.write(v and 0xff)
                            samples.write((v shr 8) and 0xff)
                        }
                    }
                }
            } catch (e: Exception) {
                running = false
                onError("Recording failed: $e")
            }
        }.also { it.start() }
        return true
    }

    /**
     * Stops recording and returns the samples normalized to [-1, 1].
     * Returns the samples captured so far even if the worker already failed
     * (see [onError]); returns null if there was no recording session.
     */
    @Synchronized
    fun stop(): FloatArray? {
        if (sessionSamples == null && thread == null && record == null) return null
        running = false
        // stop capture first so the worker's read() returns and it exits
        record?.let { runCatching { it.stop() } }
        thread?.join()
        thread = null
        record?.let {
            runCatching { it.release() }
        }
        record = null
        val samples = sessionSamples
        sessionSamples = null
        if (samples == null) return null

        val bytes = synchronized(samples) { samples.toByteArray() }
        if (bytes.size < 2) return null

        val numSamples = bytes.size / 2
        val out = FloatArray(numSamples)
        for (i in 0 until numSamples) {
            val lo = bytes[2 * i].toInt() and 0xff
            val hi = bytes[2 * i + 1].toInt()
            val v = ((hi shl 8) or lo).toShort()
            out[i] = v / 32768.0f
        }
        return out
    }
}
