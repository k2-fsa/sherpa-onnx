package com.k2fsa.sherpa.onnx.fa

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Encodes 16-bit PCM wav bytes (for saving word segments). */
object WavEncoder {
    fun encode(samples: FloatArray, sampleRate: Int): ByteArray {
        val numSamples = samples.size
        val dataSize = numSamples * 2
        val out = ByteArrayOutputStream(44 + dataSize)
        val header = ByteBuffer.allocate(44).order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray(Charsets.US_ASCII))
        header.putInt(36 + dataSize)
        header.put("WAVE".toByteArray(Charsets.US_ASCII))
        header.put("fmt ".toByteArray(Charsets.US_ASCII))
        header.putInt(16) // PCM chunk size
        header.putShort(1) // PCM format
        header.putShort(1) // mono
        header.putInt(sampleRate)
        header.putInt(sampleRate * 2) // byte rate
        header.putShort(2) // block align
        header.putShort(16) // bits per sample
        header.put("data".toByteArray(Charsets.US_ASCII))
        header.putInt(dataSize)
        out.write(header.array())

        val pcm = ByteBuffer.allocate(dataSize).order(ByteOrder.LITTLE_ENDIAN)
        for (s in samples) {
            val v = (s * 32767.0f).toInt().coerceIn(-32768, 32767)
            pcm.putShort(v.toShort())
        }
        out.write(pcm.array())
        return out.toByteArray()
    }
}
