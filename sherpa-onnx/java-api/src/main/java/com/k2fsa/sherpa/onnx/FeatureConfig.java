// Copyright 2022-2023 by zhaoming
// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class FeatureConfig {
    private final int sampleRate;
    private final int featureDim;
    private final float dither;

    private FeatureConfig(Builder builder) {
        this.sampleRate = builder.sampleRate;
        this.featureDim = builder.featureDim;
        this.dither = builder.dither;
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

   public float getDither() {
        return dither;
   }

    public static class Builder {
        private int sampleRate = 16000;
        private int featureDim = 80;
        private float dither = 0.0f;

        public FeatureConfig build() {
          if (sampleRate <= 0) {
            throw new IllegalArgumentException("sampleRate must be positive");
          }
          
          if (featureDim <= 0) {
            throw new IllegalArgumentException("featureDim must be positive");
          }
          if (dither < 0f) {
            throw new IllegalArgumentException("dither must be non-negative");
          }
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
        public Builder setDither(float dither) {
            this.dither = dither;
            return this;
        }
    }
}
