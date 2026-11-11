// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class FastClusteringConfig {
    private final int numClusters;
    private final float threshold;
    private final boolean computeConfidence;

    private FastClusteringConfig(Builder builder) {
        this.numClusters = builder.numClusters;
        this.threshold = builder.threshold;
        this.computeConfidence = builder.computeConfidence;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int getNumClusters() {
        return numClusters;
    }

    public float getThreshold() {
        return threshold;
    }

    public boolean getComputeConfidence() {
        return computeConfidence;
    }

    public static class Builder {
        private int numClusters = -1;
        private float threshold = 0.5f;
        private boolean computeConfidence = false;

        public FastClusteringConfig build() {
            return new FastClusteringConfig(this);
        }

        public Builder setNumClusters(int numClusters) {
            this.numClusters = numClusters;
            return this;
        }

        public Builder setThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder setComputeConfidence(boolean computeConfidence) {
            this.computeConfidence = computeConfidence;
            return this;
        }
    }
}
