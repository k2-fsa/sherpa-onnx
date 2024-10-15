package com.k2fsa.sherpa.onnx.speaker.diarization.screens

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri

data class WaveData(
    val sampleRate: Int? = null,
    val samples: FloatArray? = null,
    val msg: String? = null
)

// It supports only 16-bit encoded wave files
//
// References
// - https://gist.github.com/a-m-s/1991ab18fbcb0fcc2cf9
// - https://github.com/taehwandev/MediaCodecExample/blob/master/app/src/main/java/tech/thdev/mediacodecexample/audio/AACAudioDecoderThread.kt
fun readUri(context: Context, uri: Uri): WaveData {
    val extractor = MediaExtractor()
    extractor.setDataSource(context, uri, null)

    val samplesList: MutableList<FloatArray> = ArrayList()

    for (i in 0 until extractor.trackCount) {
        val format = extractor.getTrackFormat(i)
        val mime = format.getString(MediaFormat.KEY_MIME)
        if (mime?.startsWith("audio/") == true) {
            extractor.selectTrack(i)

            var encoding: Int = -1
            try {
                encoding = format.getInteger(MediaFormat.KEY_PCM_ENCODING)
            } catch (_: Exception) {
            }

            if (encoding != AudioFormat.ENCODING_PCM_16BIT) {
                return WaveData(msg = "We support only 16-bit encoded wave files")
            }

            val sampleRate = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)
            val decoder = MediaCodec.createDecoderByType(mime)
            decoder.configure(format, null, null, 0)
            decoder.start()

            val inputBuffers = decoder.inputBuffers
            var outputBuffers = decoder.outputBuffers

            val info = MediaCodec.BufferInfo()
            var eof = false

            var outputBufferIndex = -1

            while (true) {
                if (!eof) {
                    val inputBufferIndex = decoder.dequeueInputBuffer(10000)
                    if (inputBufferIndex > 0) {
                        val size = extractor.readSampleData(inputBuffers[inputBufferIndex], 0)
                        if (size < 0) {
                            decoder.queueInputBuffer(
                                inputBufferIndex,
                                0,
                                0,
                                0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM
                            )
                            eof = true
                        } else {
                            decoder.queueInputBuffer(
                                inputBufferIndex,
                                0,
                                size,
                                extractor.sampleTime,
                                0
                            )
                            extractor.advance()
                        }
                    }
                } // if (!eof)

                if (outputBufferIndex >= 0) {
                    outputBuffers[outputBufferIndex].position(0)
                }

                outputBufferIndex = decoder.dequeueOutputBuffer(info, 10000)
                if (outputBufferIndex >= 0) {
                    if (info.flags != 0) {
                        decoder.stop()
                        decoder.release()

                        var k = 0
                        for (s in samplesList) {
                            k += s.size
                        }
                        if (k == 0) {
                            return WaveData(msg = "Failed to read selected file")
                        }

                        val ans = FloatArray(k)
                        k = 0
                        for (s in samplesList) {
                            s.copyInto(ans, k)
                            k += s.size
                        }

                        return WaveData(sampleRate = sampleRate, samples = ans)
                    }

                    val buffer = outputBuffers[outputBufferIndex]
                    val chunk = ByteArray(info.size)
                    buffer[chunk]
                    buffer.clear()

                    val numSamples = info.size / 2

                    val samples = FloatArray(numSamples)
                    for (k in 0 until numSamples) {
                        // assume little endian
                        val s = chunk[2 * k] + (chunk[2 * k + 1] * 256.0f)

                        samples[k] = s / 32768.0f
                    }
                    samplesList.add(samples)

                    decoder.releaseOutputBuffer(outputBufferIndex, false)
                } else if (outputBufferIndex == MediaCodec.INFO_OUTPUT_BUFFERS_CHANGED) {
                    outputBuffers = decoder.outputBuffers
                }
            }
        }
    }

    extractor.release()
    return WaveData(msg = "not an audio file")
}