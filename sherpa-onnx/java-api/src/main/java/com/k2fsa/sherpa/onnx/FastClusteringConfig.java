// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class FastClusteringConfig {
    private final int numClusters;
    private final float threshold;

    private FastClusteringConfig(Builder builder) {
        this.numClusters = builder.numClusters;
        this.threshold = builder.threshold;
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

    public static class Builder {
        private int numClusters = -1;
        private float threshold = 0.5f;

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
    }
}
