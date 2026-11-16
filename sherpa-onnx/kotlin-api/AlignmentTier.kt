package com.k2fsa.sherpa.onnx

/**
 * One tier of an alignment: a named sequence of time intervals.
 *
 * The intervals tile the whole utterance: adjacent intervals share an
 * endpoint and silence is represented by intervals with an empty label.
 *
 * @property name tier name, e.g. "phones" or "words"
 * @property intervals the intervals of this tier, in time order; iterate
 *   them with a for-in loop
 */
data class AlignmentTier(
    val name: String,
    val intervals: List<AlignmentInterval>,
) {
    // JNI marshals the array constructor; the List is the idiomatic surface.
    constructor(name: String, intervals: Array<AlignmentInterval>) :
        this(name, intervals.asList())
}
