package com.k2fsa.sherpa.onnx.tts.engine.utils

import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import android.media.AudioTrack.PLAYSTATE_PLAYING
import android.util.Log
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import java.io.InputStream
import kotlin.coroutines.coroutineContext

class PcmAudioPlayer {
    companion object {
        private const val TAG = "AudioTrackPlayer"
    }

    private var audioTrack: AudioTrack? = null
    private var currentSampleRate = 16000

    @Suppress("DEPRECATION")
    private fun createAudioTrack(sampleRate: Int = 16000): AudioTrack {
        val mSampleRate = if (sampleRate == 0) 16000 else sampleRate
        Log.d(TAG, "createAudioTrack: sampleRate=$mSampleRate")

        val bufferSize = AudioTrack.getMinBufferSize(
            mSampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        return AudioTrack(
            AudioManager.STREAM_MUSIC,
            mSampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            bufferSize,
            AudioTrack.MODE_STREAM
        )
    }

    suspend fun play(inputStream: InputStream, sampleRate: Int = currentSampleRate) {
        val bufferSize = AudioTrack.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_OUT_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )
        inputStream.readPcmChunk(chunkSize = bufferSize) { data ->
            play(data, sampleRate)
        }
    }

    @Synchronized
    fun play(audioData: ByteArray, sampleRate: Int = currentSampleRate) {
        if (currentSampleRate == sampleRate) {
            audioTrack = audioTrack ?: createAudioTrack(sampleRate)
        } else {
            audioTrack?.stop()
            audioTrack?.release()
            audioTrack = createAudioTrack(sampleRate)
            currentSampleRate = sampleRate
        }

        if (audioTrack!!.playState != PLAYSTATE_PLAYING) audioTrack!!.play()

        audioTrack!!.write(audioData, 0, audioData.size)
        println("play done..")
    }


    fun stop() {
        audioTrack?.stop()
    }

    fun release() {
        audioTrack?.release()
    }

    suspend fun InputStream.readPcmChunk(
        bufferSize: Int = 4096,
        chunkSize: Int = 2048,
        onRead: suspend (ByteArray) -> Unit
    ) {
        var bufferFilledCount = 0
        val buffer = ByteArray(bufferSize)

        while (coroutineContext.isActive) {
            val readLen =
                this.read(buffer, bufferFilledCount, chunkSize - bufferFilledCount)
            if (readLen == -1) {
                if (bufferFilledCount > 0) {
                    val chunkData = buffer.copyOfRange(0, bufferFilledCount)
                    onRead.invoke(chunkData)
                }
                break
            }
            if (readLen == 0) {
                delay(100)
                continue
            }

            bufferFilledCount += readLen
            if (bufferFilledCount >= chunkSize) {
                val chunkData = buffer.copyOfRange(0, chunkSize)

                onRead.invoke(chunkData)
                bufferFilledCount = 0
            }
        }
    }
}