// Copyright (c)  2026  Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

/**
 * One time interval of an alignment tier.
 *
 * <p>For example, the word "ask" of a 3.85 s utterance:
 *
 * <pre>
 * AlignmentInterval(start=0.06, end=0.45, text="ask")
 * </pre>
 */
public final class AlignmentInterval {
    private final float start;
    private final float end;
    private final String text;

    /**
     * Constructor. Called from JNI; applications normally only read the
     * objects returned by {@link ForcedAligner#compute}.
     *
     * @param start start time of this interval in seconds
     * @param end end time of this interval in seconds
     * @param text label of this interval: a phone, a word/character, or
     *     empty for silence gaps
     */
    public AlignmentInterval(float start, float end, String text) {
        this.start = start;
        this.end = end;
        this.text = text;
    }

    /** Returns the start time of this interval in seconds. */
    public float getStart() {
        return start;
    }

    /** Returns the end time of this interval in seconds. */
    public float getEnd() {
        return end;
    }

    /**
     * Returns the label of this interval: a phone, a word/character, or
     * empty for silence gaps.
     */
    public String getText() {
        return text;
    }

    @Override
    public String toString() {
        return "AlignmentInterval(" + start + ", " + end + ", \"" + text + "\")";
    }
}
