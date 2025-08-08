// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class SileroVadModelConfig {
    private final String model;
    private final float threshold;
    private final float minSilenceDuration;
    private final float minSpeechDuration;
    private final int windowSize;
    private final float maxSpeechDuration;

    private SileroVadModelConfig(Builder builder) {
        this.model = builder.model;
        this.threshold = builder.threshold;
        this.minSilenceDuration = builder.minSilenceDuration;
        this.minSpeechDuration = builder.minSpeechDuration;
        this.windowSize = builder.windowSize;
        this.maxSpeechDuration = builder.maxSpeechDuration;
    }

    public static Builder builder() {
        return new Builder();
    }

    public String getModel() {
        return model;
    }

    public float getThreshold() {
        return threshold;
    }

    public float getMinSilenceDuration() {
        return minSilenceDuration;
    }

    public float getMinSpeechDuration() {
        return minSpeechDuration;
    }

    public int getWindowSize() {
        return windowSize;
    }

    public float getMaxSpeechDuration() {
        return maxSpeechDuration;
    }

    public static class Builder {
        private String model = "";
        private float threshold = 0.5f;
        private float minSilenceDuration = 0.25f;
        private float minSpeechDuration = 0.5f;
        private int windowSize = 512;
        private float maxSpeechDuration = 5.0f;

        public SileroVadModelConfig build() {
            return new SileroVadModelConfig(this);
        }


        public Builder setModel(String model) {
            this.model = model;
            return this;
        }

        public Builder setThreshold(float threshold) {
            this.threshold = threshold;
            return this;
        }

        public Builder setMinSilenceDuration(float minSilenceDuration) {
            this.minSilenceDuration = minSilenceDuration;
            return this;
        }

        public Builder setMinSpeechDuration(float minSpeechDuration) {
            this.minSpeechDuration = minSpeechDuration;
            return this;
        }

        public Builder setWindowSize(int windowSize) {
            this.windowSize = windowSize;
            return this;
        }

        public Builder setMaxSpeechDuration(float maxSpeechDuration) {
            this.maxSpeechDuration = maxSpeechDuration;
            return this;
        }
    }
}
