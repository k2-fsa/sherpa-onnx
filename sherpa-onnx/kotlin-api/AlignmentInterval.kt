package com.k2fsa.sherpa.onnx

/**
 * One time interval of an alignment tier.
 *
 * For example, the word "ask" of a 3.85 s utterance:
 *
 * ```
 * AlignmentInterval(start=0.06, end=0.45, text="ask")
 * ```
 *
 * @property start start time of this interval in seconds
 * @property end end time of this interval in seconds
 * @property text label of this interval: the phone symbol in the "phones"
 *   tier or the word/character in the "words" tier; empty for a silence gap
 *   in the "words" tier (the intervals tile the whole utterance, so the
 *   gaps between words are reported as intervals with empty text)
 */
data class AlignmentInterval(
    val start: Float,
    val end: Float,
    val text: String,
)
