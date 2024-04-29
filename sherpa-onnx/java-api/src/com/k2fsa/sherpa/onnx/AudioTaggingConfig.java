// Copyright 2024 Xiaomi Corporation

package com.k2fsa.sherpa.onnx;

public class AudioTaggingConfig {
    private final AudioTaggingModelConfig model;
    private final String labels;
    private final int topK;

    private AudioTaggingConfig(Builder builder) {
        this.model = builder.model;
        this.labels = builder.labels;
        this.topK = builder.topK;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder {
        private AudioTaggingModelConfig model = AudioTaggingModelConfig.builder().build();
        private String labels = "";
        private int topK = 5;

        public AudioTaggingConfig build() {
            return new AudioTaggingConfig(this);
        }

        public Builder setModel(AudioTaggingModelConfig model) {
            this.model = model;
            return this;
        }

        public Builder setLabels(String labels) {
            this.labels = labels;
            return this;
        }

        public Builder setTopK(int topK) {
            this.topK = topK;
            return this;
        }
    }
}
