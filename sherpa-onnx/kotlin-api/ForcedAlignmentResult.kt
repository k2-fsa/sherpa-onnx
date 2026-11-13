package com.k2fsa.sherpa.onnx

/**
 * Result of [ForcedAligner.compute]: the phone and word boundaries of one
 * utterance, and the same content as a Praat TextGrid.
 *
 * ```
 * for (tier in result.tiers) {          // "phones", then "words"
 *     for (interval in tier.intervals) {
 *         println("${interval.start} - ${interval.end}  ${interval.text}")
 *     }
 * }
 * ```
 *
 * @property duration duration of the aligned utterance in seconds
 * @property tiers the alignment tiers: `tiers[0]` is the "phones" tier,
 *   `tiers[1]` the "words" tier. The result is empty (no tiers) when the
 *   alignment failed; see [ForcedAligner.compute]
 * @property textGrid the alignment as a Praat TextGrid string (long format).
 *   Write it to a `.TextGrid` file and open it with Praat
 *   (https://www.fon.hum.uva.nl/praat/) to visualize the alignment (or use
 *   any other TextGrid reader)
 */
data class ForcedAlignmentResult(
    val duration: Float,
    val tiers: List<AlignmentTier>,
    val textGrid: String,
) {
    // JNI marshals the array constructor; the List is the idiomatic surface.
    constructor(duration: Float, tiers: Array<AlignmentTier>, textGrid: String) :
        this(duration, tiers.asList(), textGrid)
}
