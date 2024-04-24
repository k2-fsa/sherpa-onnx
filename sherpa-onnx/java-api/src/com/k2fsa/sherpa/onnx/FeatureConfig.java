// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class FeatureConfig {
    private final int sampleRate;
    private final int featureDim;

    private FeatureConfig(Builder builder) {
        this.sampleRate = builder.sampleRate;
        this.featureDim = builder.featureDim;
    }

    public static Builder builder() {
        return new Builder();
    }

    public int getSampleRate() {
        return sampleRate;
    }

    public int getFeatureDim() {
        return featureDim;
    }

    public static class Builder {
        private int sampleRate = 16000;
        private int featureDim = 80;

        public FeatureConfig build() {
            return new FeatureConfig(this);
        }

        public Builder setSampleRate(int sampleRate) {
            this.sampleRate = sampleRate;
            return this;
        }

        public Builder setFeatureDim(int featureDim) {
            this.featureDim = featureDim;
            return this;
        }
    }
}
