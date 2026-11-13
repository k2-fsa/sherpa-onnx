// Copyright (c)  2026  Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

/**
 * Result of {@link ForcedAligner#compute}: the phone and word boundaries of
 * one utterance, and the same content as a Praat TextGrid.
 *
 * <pre>
 * for (AlignmentTier tier : result.getTiers()) {
 *     for (AlignmentInterval interval : tier.getIntervals()) {
 *         System.out.printf("%.2f - %.2f  %s%n", interval.getStart(),
 *                 interval.getEnd(), interval.getText());
 *     }
 * }
 * </pre>
 */
public final class ForcedAlignmentResult {
    private final float duration;
    private final AlignmentTier[] tiers;
    private final String textGrid;

    /**
     * Constructor. Called from JNI; applications normally only read the
     * objects returned by {@link ForcedAligner#compute}.
     *
     * @param duration duration of the aligned utterance in seconds
     * @param tiers the alignment tiers: {@code tiers[0]} is the
     *     {@code "phones"} tier, {@code tiers[1]} the {@code "words"} tier.
     *     The result is empty (no tiers) when the alignment failed; see
     *     {@link ForcedAligner#compute}
     * @param textGrid the alignment as a Praat TextGrid string (long format)
     */
    public ForcedAlignmentResult(float duration, AlignmentTier[] tiers,
                                 String textGrid) {
        this.duration = duration;
        this.tiers = tiers;
        this.textGrid = textGrid;
    }

    /** Returns the duration of the aligned utterance in seconds. */
    public float getDuration() {
        return duration;
    }

    /**
     * Returns the alignment tiers. {@code tiers[0]} is the {@code "phones"}
     * tier and {@code tiers[1]} the {@code "words"} tier; iterate them with
     * a for-each loop. The result is empty (no tiers) when the alignment
     * failed; see {@link ForcedAligner#compute}.
     */
    public List<AlignmentTier> getTiers() {
        return Collections.unmodifiableList(Arrays.asList(tiers));
    }

    /**
     * Returns the alignment as a Praat TextGrid string (long format). Write
     * it to a {@code .TextGrid} file and open it with Praat
     * (https://www.fon.hum.uva.nl/praat/) to visualize the alignment (or
     * use any other TextGrid reader).
     */
    public String toTextGrid() {
        return textGrid;
    }

    @Override
    public String toString() {
        return toTextGrid();
    }
}
