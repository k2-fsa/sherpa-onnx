/*
 * // Copyright 2022-2023 by zhaoming
 */

package com.k2fsa.sherpaonnx.rcglib;

public class EndpointRule {
    final private boolean mustContainNonSilence;
    final private float minTrailingSilence;
    final private float minUtteranceLength;

    public EndpointRule(boolean mustContainNonSilence, float minTrailingSilence, float minUtteranceLength) {
        this.mustContainNonSilence = mustContainNonSilence;
        this.minTrailingSilence = minTrailingSilence;
        this.minUtteranceLength = minUtteranceLength;
    }


    public float getMinTrailingSilence() {
        return minTrailingSilence;
    }

    public float getMinUtteranceLength() {
        return minUtteranceLength;
    }

    public boolean getMustContainNonSilence() {
        return mustContainNonSilence;
    }
}
