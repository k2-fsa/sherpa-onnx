// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation
package com.k2fsa.sherpa.onnx;

public class EndpointRule {

    private final boolean mustContainNonSilence;
    private final float minTrailingSilence;
    private final float minUtteranceLength;

    private EndpointRule(Builder builder) {
        this.mustContainNonSilence = builder.mustContainNonSilence;
        this.minTrailingSilence = builder.minTrailingSilence;
        this.minUtteranceLength = builder.minUtteranceLength;
    }

    public static Builder builder() {
        return new Builder();
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

    public static class Builder {
        private boolean mustContainNonSilence = false;
        private float minTrailingSilence = 0;
        private float minUtteranceLength = 0;

        public EndpointRule build() {
            return new EndpointRule(this);
        }

        public Builder setMustContainNonSilence(boolean mustContainNonSilence) {
            this.mustContainNonSilence = mustContainNonSilence;
            return this;
        }

        public Builder setMinTrailingSilence(float minTrailingSilence) {
            this.minTrailingSilence = minTrailingSilence;
            return this;
        }

        public Builder setMinUtteranceLength(float minUtteranceLength) {
            this.minUtteranceLength = minUtteranceLength;
            return this;
        }
    }
}