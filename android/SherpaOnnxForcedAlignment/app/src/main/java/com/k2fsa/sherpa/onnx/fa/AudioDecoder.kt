package com.k2fsa.sherpa.onnx.fa

import com.antonkarpenko.ffmpegkit.FFmpegKit
import com.antonkarpenko.ffmpegkit.ReturnCode
import java.io.File
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Result of decoding an audio file. */
data class DecodedAudio(
    val samples: FloatArray, // float32, normalized to [-1, 1]
    val sampleRate: Int,
    val duration: Double, // seconds
)

/**
 * Decodes any audio/video format to 16 kHz mono Float32 PCM with FFmpegKit.
 *
 * Mirrors vad-from-file/lib/audio_decoder_native.dart (the same FFmpeg
 * command); the sherpa-onnx aligner wants exactly this input.
 */
object AudioDecoder {
    const val SAMPLE_RATE = 16000

    /**
     * Decodes [input] to 16 kHz mono float32 PCM. Returns null when the file
     * cannot be decoded. [cacheDir] is used for the temporary raw output.
     */
    fun decode(input: File, cacheDir: File): DecodedAudio? {
        val output = File(cacheDir, "fa_decoded_${System.nanoTime()}.raw")
        return try {
            val session = FFmpegKit.execute(
                "-nostdin -hide_banner -loglevel error " +
                    "-i \"${input.path}\" -ar $SAMPLE_RATE -ac 1 " +
                    "-f f32le -acodec pcm_f32le -y \"${output.path}\""
            )
            if (!ReturnCode.isSuccess(session.returnCode)) {
                return null
            }
            // f32le: 4 bytes per sample
            val outputBytes = output.length()
            if (outputBytes < 4 || outputBytes / 4 > Int.MAX_VALUE) {
                return null
            }
            val numSamples = (outputBytes / 4).toInt()
            // One allocation for the samples; read the raw file in chunks so
            // we never hold a full byte copy of it in memory.
            val samples = FloatArray(numSamples)
            val chunk = ByteArray(256 * 1024)
            FileInputStream(output).use { ins ->
                var filled = 0
                while (filled < numSamples) {
                    val want = minOf(chunk.size, (numSamples - filled) * 4)
                    var got = 0
                    while (got < want) {
                        val n = ins.read(chunk, got, want - got)
                        if (n < 0) break
                        got += n
                    }
                    if (got < want) {
                        return null // truncated raw output
                    }
                    val floats = ByteBuffer.wrap(chunk, 0, want)
                        .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
                    floats.get(samples, filled, want / 4)
                    filled += want / 4
                }
            }
            DecodedAudio(
                samples = samples,
                sampleRate = SAMPLE_RATE,
                duration = numSamples.toDouble() / SAMPLE_RATE,
            )
        } catch (e: Throwable) {
            // includes OutOfMemoryError on very long inputs
            null
        } finally {
            output.delete()
        }
    }
}
