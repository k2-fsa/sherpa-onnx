package com.k2fsa.sherpa.onnx.fa

import com.antonkarpenko.ffmpegkit.FFmpegKit
import com.antonkarpenko.ffmpegkit.ReturnCode
import java.io.File
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
            val bytes = output.readBytes()
            if (bytes.size < 4) {
                return null
            }
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
            val samples = FloatArray(buffer.remaining())
            buffer.get(samples)
            DecodedAudio(
                samples = samples,
                sampleRate = SAMPLE_RATE,
                duration = samples.size.toDouble() / SAMPLE_RATE,
            )
        } catch (e: Exception) {
            null
        } finally {
            output.delete()
        }
    }
}
