package com.k2fsa.sherpa.onnx

import android.content.res.AssetManager

/**
 * Forced aligner: given a wave file and its transcript, it computes the
 * start/end time of every phone and word.
 *
 * ```
 * val config = ForcedAlignerConfig(
 *     modelDir = "/path/to/english-mfa-v3.1.0",
 *     lexicon = "/path/to/english-mfa-v3.1.0/lexicon.txt",
 * )
 *
 * val aligner = ForcedAligner(config = config)
 * val stream = aligner.createStream()
 * stream.acceptWaveform(samples, sampleRate)  // float32, normalized to [-1, 1]
 * val result = aligner.compute(stream, text)
 * println(result.textGrid)
 * ```
 *
 * @param assetManager Android asset manager to load the model from the APK
 *   assets (modelDir and lexicon are then asset paths); null to load from
 *   the filesystem
 * @param config configuration for forced alignment
 */
class ForcedAligner(
    assetManager: AssetManager? = null,
    config: ForcedAlignerConfig,
) {
    private var ptr: Long

    init {
        ptr = if (assetManager != null) {
            newFromAsset(assetManager, config)
        } else {
            newFromFile(config)
        }
        require(ptr != 0L) {
            "Invalid ForcedAlignerConfig: failed to create native ForcedAligner"
        }
    }

    protected fun finalize() {
        if (ptr != 0L) {
            delete(ptr)
            ptr = 0
        }
    }

    // You'd better call it manually if it is not used anymore
    fun release() = finalize()

    /**
     * Creates an utterance stream. Accept a waveform with
     * [OfflineStream.acceptWaveform], then pass the stream to [compute].
     */
    fun createStream(): OfflineStream {
        val p = createStream(ptr)
        return OfflineStream(p)
    }

    /**
     * Computes the alignment of one utterance.
     *
     * @param stream a stream created by [createStream] and filled with a
     *   waveform
     * @param text the transcript of the whole utterance. Hanzi text is
     *   tokenized per character (multi-character dictionary keys give
     *   pronunciation context); anything else is split on whitespace.
     *   Punctuation is ignored
     * @return the alignment result; empty (no tiers) with a warning on
     *   stderr when the stream is null, the text is empty, the stream has no
     *   features, or word boundaries cannot be recovered
     */
    fun compute(stream: OfflineStream, text: String): ForcedAlignmentResult =
        compute(ptr, stream.ptr, text)

    private external fun delete(ptr: Long)

    private external fun newFromFile(config: ForcedAlignerConfig): Long

    private external fun newFromAsset(
        assetManager: AssetManager,
        config: ForcedAlignerConfig,
    ): Long

    private external fun createStream(ptr: Long): Long

    private external fun compute(
        ptr: Long,
        streamPtr: Long,
        text: String,
    ): ForcedAlignmentResult

    companion object {
        init {
            System.loadLibrary("sherpa-onnx-jni")
        }
    }
}
