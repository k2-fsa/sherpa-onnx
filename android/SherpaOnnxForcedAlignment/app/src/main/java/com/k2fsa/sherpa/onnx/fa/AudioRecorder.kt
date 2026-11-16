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
 * Usage: [start], then [stop] to get the recorded samples.
 */
class AudioRecorder {
    companion object {
        const val SAMPLE_RATE = 16000
    }

    private var record: AudioRecord? = null
    private var thread: Thread? = null
    private val samples = ByteArrayOutputStream()
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
        samples.reset()
        record = rec
        running = true
        thread = Thread {
            val buf = ShortArray(2048)
            try {
                rec.startRecording()
                while (running) {
                    val n = rec.read(buf, 0, buf.size)
                    if (n <= 0) break
                    synchronized(samples) {
                        for (i in 0 until n) {
                            val v = buf[i].toInt()
                            samples.write(v and 0xff)
                            samples.write((v shr 8) and 0xff)
                        }
                    }
                }
            } catch (e: Exception) {
                // recording failed; stop() reports the samples read so far
            }
        }.also { it.start() }
        return true
    }

    /**
     * Stops recording and returns the samples normalized to [-1, 1].
     * Returns null if nothing was recorded.
     */
    @Synchronized
    fun stop(): FloatArray? {
        if (!running) return null
        running = false
        thread?.join(1000)
        thread = null
        record?.let {
            runCatching { it.stop() }
            runCatching { it.release() }
        }
        record = null

        val bytes: ByteArray
        synchronized(samples) { bytes = samples.toByteArray() }
        samples.reset()
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
