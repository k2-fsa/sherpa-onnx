// Copyright (c)  2026  Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * One tier of an alignment: a named sequence of time intervals.
 *
 * <p>The intervals tile the whole utterance: adjacent intervals share an
 * endpoint and silence is represented by intervals with an empty label.
 */
public final class AlignmentTier {
    private final String name;
    private final AlignmentInterval[] intervals;

    /**
     * Constructor. Called from JNI; applications normally only read the
     * objects returned by {@link ForcedAligner#compute}.
     *
     * @param name tier name, e.g. {@code "phones"} or {@code "words"}
     * @param intervals the intervals of this tier, in time order
     */
    public AlignmentTier(String name, AlignmentInterval[] intervals) {
        this.name = name;
        this.intervals = intervals;
    }

    /** Returns the tier name, e.g. {@code "phones"} or {@code "words"}. */
    public String getName() {
        return name;
    }

    /**
     * Returns the intervals of this tier in time order; iterate them with a
     * for-each loop.
     */
    public List<AlignmentInterval> getIntervals() {
        return Collections.unmodifiableList(Arrays.asList(intervals));
    }

    @Override
    public String toString() {
        return "AlignmentTier(" + name + ", " + intervals.length + " intervals)";
    }
}
