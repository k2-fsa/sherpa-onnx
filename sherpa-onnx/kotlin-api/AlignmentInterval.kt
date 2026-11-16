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
 * @property text label of this interval: a phone, a word/character, or empty
 *   for silence gaps
 */
data class AlignmentInterval(
    val start: Float,
    val end: Float,
    val text: String,
)
