package com.k2fsa.sherpa.onnx.fa

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack

/**
 * Plays the audio of one aligned word: a slice of the decoded mono PCM with
 * sample accuracy.
 *
 * It plays exactly the samples the aligner saw, so a word's audio matches
 * its [start, end] timestamps and never includes the neighbors — unlike
 * seeking a media player, which is approximate for compressed formats.
 */
class WordPlayer {
    private var track: AudioTrack? = null

    /**
     * Plays samples in [startSeconds, endSeconds). Calls [onDone] on the
     * AudioTrack event thread when the word has finished playing; it is not
     * called when [stop] is used or when another [play] supersedes this one.
     */
    @Synchronized
    fun play(
        samples: FloatArray,
        sampleRate: Int,
        startSeconds: Float,
        endSeconds: Float,
        onDone: () -> Unit,
    ) {
        stopInternal()
        val start = (startSeconds * sampleRate).toInt().coerceIn(0, samples.size)
        val end = (endSeconds * sampleRate).toInt().coerceIn(start, samples.size)
        val numSamples = end - start
        if (numSamples <= 0) {
            onDone()
            return
        }

        val pcm = ShortArray(numSamples)
        for (i in 0 until numSamples) {
            val v = (samples[start + i] * 32767.0f).toInt().coerceIn(-32768, 32767)
            pcm[i] = v.toShort()
        }

        val minBuf = AudioTrack.getMinBufferSize(
            sampleRate, AudioFormat.CHANNEL_OUT_MONO, AudioFormat.ENCODING_PCM_16BIT
        )
        val t = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                    .setSampleRate(sampleRate)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setBufferSizeInBytes(maxOf(minBuf * 2, numSamples * 2))
            .setTransferMode(AudioTrack.MODE_STATIC)
            .build()
        t.write(pcm, 0, numSamples)

        // Fire onDone when the playback head reaches the end of the slice.
        t.setNotificationMarkerPosition(numSamples)
        t.setPlaybackPositionUpdateListener(
            object : AudioTrack.OnPlaybackPositionUpdateListener {
                override fun onMarkerReached(track: AudioTrack?) {
                    synchronized(this@WordPlayer) {
                        // ignore if this playback was stopped or superseded
                        if (this@WordPlayer.track !== t) return
                        this@WordPlayer.track = null
                    }
                    runCatching { t.stop() }
                    runCatching { t.release() }
                    onDone()
                }

                override fun onPeriodicNotification(track: AudioTrack?) = Unit
            }
        )
        track = t
        t.play()
    }

    /** Stops the current word; onDone is not called. */
    @Synchronized
    fun stop() {
        stopInternal()
    }

    private fun stopInternal() {
        track?.let { t ->
            runCatching { t.stop() }
            runCatching { t.release() }
        }
        track = null
    }
}
